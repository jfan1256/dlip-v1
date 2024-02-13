import io
import torch
import json
import torch.optim as optim
import torch.multiprocessing as mp

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from azure.storage.blob import BlobServiceClient

from class_data.azure_blob import AzureBlob
from class_train_help.metric_log import SmoothedValue, MetricLogger
from class_transform.randaugment import RandomAugment
from class_model.dlip_caption import DLIPCaption
from class_data.preprocess import Preprocess
from class_data.image_tensor import ImageTensor
from class_train_help.ddp_handle import DDPHandle
from class_train_help.metric_log import warmup_lr_schedule, step_lr_schedule
from utils.system import *

class DLIPCaptionTrain:
    def __init__(self,
                 azure=None,
                 connection_string=None,
                 container_name=None,
                 multi=None,
                 ddp_server=None,
                 image_size=None,
                 batch_size=None,
                 world_size=None,
                 partial=None,
                 val_split=None,
                 device=None,
                 scheduler=None,
                 warmup_steps=None,
                 warmup_lr=None,
                 min_lr=None,
                 lr_decay_rate=None,
                 dlip_bert=None,
                 dlip_vit=None,
                 accumulate=None,
                 learning_rate=None,
                 weight_decay=None,
                 dlip_epoch=None,
                 checkpoint_epoch=None,
                 num_epoch=None,
                 freeze=None,
                 gradient_clip=None,
                 ):

        '''
        azure (bool_str): Azure Cloud training or not
        connection_string (str): Connection string for Azure Blob Storage
        container_name (str): Name of the container in Azure Blob Storage
        multi (bool_str): Multi-GPU training or not
        ddp_server (str): Pytorch DDP Server (i.e., 'nccl' for Linux, 'gloo' for Windows)
        image_size (int): Image size
        batch_size (int): Batch size
        world_size (int): Number of GPUs
        partial (bool_str): Training on partial data or not (set to 1000 datapoints)
        val_split (float): Fraction of data to allocate for validation
        device (str): Training device
        scheduler (bool_str): Learning rate scheduler or not
        warmup_steps : Number of warmup steps (number of batches) for lr scheduler
        warmup_lr : Warmup learning rate for lr scheduler
        min_lr : Minimum linear rate for lr scheduler
        lr_decay_rate" : Linear decay rate for lr scheduler
        dlip_bert (str): Size of BERT model (either 'tiny', 'base', or 'large')
        dlip_vit (str): Size of Vit model (either 'tiny', 'base', or 'large')
        accumulate (int): Number of batches to accumulate before optimizer.step()
        learning_rate (float): Learning rate
        weight_decay (float): Learning rate decay
        dlip_epoch (int): Epoch number for DLIP model
        checkpoint_epoch (int): Checkpoint epoch number for continuous training (i.e., continue training after model stopped training)
        num_epoch (int): Number of epochs to train
        freeze (bool_str): Freeze weights for training or not (to test whether the model is messing up or not)
        gradient_clip (bool_str): Gradient clipping or not (helpful for ita loss)
        '''

        self.azure = azure
        self.connection_string = connection_string
        self.container_name = container_name
        self.multi = multi
        self.ddp_server = ddp_server
        self.image_size = image_size
        self.batch_size = batch_size
        self.world_size = world_size
        self.partial = partial
        self.val_split = val_split
        self.device = device
        self.scheduler = scheduler
        self.warmup_steps=warmup_steps
        self.warmup_lr=warmup_lr
        self.min_lr=min_lr
        self.lr_decay_rate=lr_decay_rate
        self.dlip_bert = dlip_bert
        self.dlip_vit = dlip_vit
        self.accumulate = accumulate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dlip_epoch = dlip_epoch
        self.checkpoint_epoch = checkpoint_epoch
        self.num_epoch = num_epoch
        self.freeze = freeze
        self.gradient_clip = gradient_clip

    def train(self, rank):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------DISTRIBUTED DATA PARALLEL SETUP----------------------------------------------------------------------------
        print("-"*60 + "\nSetting up DDP...")
        # Setup MultiGPU
        multi_gpu = DDPHandle

        # Set up the process groups for multi-gpu
        if self.multi == "True":
            multi_gpu.setup(rank, self.world_size, self.ddp_server)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------LOAD DATA--------------------------------------------------------------------------------------
        print("-"*60 + "\nLoading data...")
        # Load Data
        if self.azure == "True":
            file_pattern = r'all_data_tokenize_\d+\.parquet\.brotli'
            all_data = AzureBlob(connection_string=self.connection_string, container_name=self.container_name, file_pattern=file_pattern)._concat_files()
        else:
            file_pattern = 'all_data_tokenize_*.parquet.brotli'
            folder_path = get_data() / 'all' / 'chunks'
            all_data = Preprocess(folder_path=folder_path, file_pattern=file_pattern)._concat_files()

        # Train on partial data
        if self.partial > 0:
            all_data = all_data.sample(n=self.partial, random_state=42)

        # Split data into Train/Val
        val_data = all_data.sample(frac=self.val_split, random_state=42)
        train_data = all_data.drop(val_data.index)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------CREATE DATALOADER-----------------------------------------------------------------------------------
        print("-"*60 + "\nCreating dataloader...")
        # Create transformation
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.image_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])

        # Create dataloader
        dataset_train = ImageTensor(azure=self.azure, connection_string=self.connection_string, container_name=self.container_name, data=train_data, image_column='image_name', caption_column='caption', transform=transform_train, max_words=30)
        dataset_val = ImageTensor(azure=self.azure, connection_string=self.connection_string, container_name=self.container_name, data=val_data, image_column='image_name', caption_column='caption', transform=transform_train, max_words=30)
        if self.multi == "True":
            dataloader_train = multi_gpu.prepare(dataset_train, rank, self.world_size, self.batch_size, shuffle=False)
        else:
            dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False, drop_last=True)
            dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, drop_last=True)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------CREATE MODEL-----------------------------------------------------------------------------------
        print("-"*60 + "\nCreating model...")
        # Create model
        if self.multi == "True":
            dlip_caption_train = DLIPCaption(image_size=self.image_size, bert=self.dlip_bert, vit=self.dlip_vit, multi=True).to(torch.device(f'cuda:{rank}'))
            dlip_caption_train = DDP(dlip_caption_train, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        else:
            dlip_caption_train = DLIPCaption(image_size=self.image_size, bert=self.dlip_bert, vit=self.dlip_vit, multi=False).to(torch.device(self.device))

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------LOAD MODEL-------------------------------------------------------------------------------------
        if self.dlip_epoch > 0:
            print("-"*60 + "\nLoading model...")
            # Load checkpoint
            checkpoint_path = get_data() / 'dlip'
            checkpoint_files = [f for f in checkpoint_path.glob('*.pt')]

            checkpoints = {}
            for file in checkpoint_files:
                epoch_number = file.stem.split('_')[-1]
                if epoch_number == self.checkpoint_epoch:
                    checkpoints[epoch_number] = torch.load(file, map_location=self.device)

            state_dict = checkpoints[self.checkpoint_epoch]['state_dict']

            # Retrieve only caption weights
            include_components = ['image_encoder', 'text_decoder', 'image_text_proj']
            exclude_components = ['text_encoder', 'image_proj', 'text_proj', 'itm_head',
                                  'hr_retrieval_image', 'hr_retrieval_text', 'hr_caption_image',
                                  'hr_caption_text', 'image_encoder_m', 'image_proj_m',
                                  'text_encoder_m', 'text_proj_m']

            state_dict = {name: param for name, param in state_dict.items()
                                   if any(include_comp in name for include_comp in include_components)
                                   and not any(exclude_comp in name for exclude_comp in exclude_components)}

            # Load model
            dlip_caption_train.load_state_dict(state_dict)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------CREATE OPTIMIZER/CHECKPOINT--------------------------------------------------------------------------
        print("-"*60 + "\nCreating optimizer...")
        # Create optimizer
        optimizer = optim.Adam(dlip_caption_train.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Handle checkpoint
        if self.azure == "True":
            if self.checkpoint_epoch > 0:
                # Create a blob client using the local file name as the name for the blob
                blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                container_client = blob_service_client.get_container_client(self.container_name)

                checkpoint_blob_name = f'dlip_caption/epoch_{self.checkpoint_epoch + 1}.pt'
                blob_client = container_client.get_blob_client(blob=checkpoint_blob_name)

                # Download the blob content to a stream
                checkpoint_stream = blob_client.download_blob().readall()

                # Load checkpoint from the stream
                checkpoint = torch.load(io.BytesIO(checkpoint_stream))
                dlip_caption_train.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            if self.checkpoint_epoch > 0:
                # Load checkpoint model and optimizer
                checkpoint_path = get_data() / 'dlip_caption' / f'epoch_{self.checkpoint_epoch + 1}.pt'
                checkpoint = torch.load(checkpoint_path)
                dlip_caption_train.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------START TRAINING---------------------------------------------------------------------------------
        print("-"*60 + "\nStarting train...")
        # Store losses
        train_losses_caption = []

        # Create Metric Loggers
        train_metric_logger = MetricLogger(delimiter="  ")
        train_metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        train_metric_logger.add_meter('train_loss_caption', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        # Iterate through num_epoch
        for epoch in range(self.checkpoint_epoch, self.num_epoch):
            # Train
            dlip_caption_train.train()

            # Step linear scheduler
            if self.scheduler == "True":
                step_lr_schedule(optimizer, epoch, self.learning_rate, self.min_lr, self.lr_decay_rate)

            # Freeze All Weights
            if self.freeze == "True":
                for param in dlip_caption_train.parameters():
                    param.requires_grad = False

            # Print Epoch
            header = 'Train Epoch: [{}]'.format(epoch+1)
            print_freq = 10

            # Initialize or Reset Losses for Each Epoch
            num_batches = 0

            for i, (image, caption, idx) in enumerate(train_metric_logger.log_every(dataloader_train, print_freq, header)):
                # Warm up lr scheduler
                if epoch == 0 and self.scheduler == "True":
                    warmup_lr_schedule(optimizer, i, self.warmup_steps, self.warmup_lr, self.learning_rate)

                # Zero out gradients
                if self.freeze != "True":
                    optimizer.zero_grad()

                # Put data to device
                if self.multi == "True":
                    image = image.to(torch.device(f'cuda:{rank}'))
                else:
                    image = image.to(torch.device(self.device))

                # Increment Batch Count
                num_batches += 1

                # Retrieve Losses from Dlip Class
                train_loss_lm = dlip_caption_train(image, caption)
                train_loss = train_loss_lm

                # Perform Optimization
                if self.gradient_clip == "True":
                    # Gradient Clipping
                    # Average across accumulation
                    if self.accumulate != 0:
                        train_loss /= self.accumulate

                    train_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(dlip_caption_train.parameters(), 0.02)

                    # Accumulate gradient
                    if self.accumulate != 0 and num_batches%self.accumulate == 0:
                        print(f"Accumulating gradient step: num_batches = {num_batches}...")
                        optimizer.step()
                    elif self.accumulate == 0:
                        optimizer.step()

                else:
                    # Average across accumulation
                    if self.accumulate != 0:
                        train_loss /= self.accumulate

                    train_loss.backward()

                    # Accumulate gradient
                    if self.accumulate != 0 and num_batches%self.accumulate == 0:
                        print(f"Accumulating gradient step: num_batches = {num_batches}...")
                        optimizer.step()
                    elif self.accumulate == 0:
                        optimizer.step()

                # Store Losses
                train_losses_caption.append(train_loss_lm.item())

                # Log Losses
                train_metric_logger.update(train_loss_caption = train_loss_lm.item())
                train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # Save Model Checkpoint
            if self.multi == "True":
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': dlip_caption_train.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
            else:
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': dlip_caption_train.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

            if self.azure == "True":
                # Create a blob client
                blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                container_client = blob_service_client.get_container_client(self.container_name)

                # Name of the blob (file in Azure)
                blob_name = f'dlip_caption/epoch_{epoch + 1}.pt'

                # Create a BlobClient
                blob_client = container_client.get_blob_client(blob=blob_name)

                # Convert the checkpoint dictionary to a byte stream
                checkpoint_stream = io.BytesIO()
                torch.save(checkpoint, checkpoint_stream)
                checkpoint_stream.seek(0)

                # Upload the stream to Azure Blob Storage
                blob_client.upload_blob(checkpoint_stream, overwrite=True)
            else:
                checkpoint_path = get_data() / 'dlip_caption' / f'epoch_{epoch + 1}.pt'
                os.makedirs(checkpoint_path.parent, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)

            # Save Losses
            losses_dict = {
                "train_losses_caption": train_losses_caption,
            }

            if self.azure == "True":
                json_string = json.dumps(losses_dict)
                bytes_stream = io.BytesIO(json_string.encode())

                # Name of the blob (file in Azure)
                blob_name = f'log/losses_caption.json'

                # Initialize the Blob Service Client
                blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                container_client = blob_service_client.get_container_client(container=self.container_name)

                # Create a BlobClient
                blob_client = container_client.get_blob_client(blob=blob_name)

                # Upload the bytes stream to Azure Blob Storage
                bytes_stream.seek(0)
                blob_client.upload_blob(bytes_stream, overwrite=True)
            else:
                # Save file
                log_file_path = get_data() / 'log' / 'losses_caption.json'
                with open(log_file_path, 'w') as file:
                    json.dump(losses_dict, file)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------CLEAN UP------------------------------------------------------------------------------------
        # Clean up multi-gpu process
        if self.multi == "True":
            multi_gpu.cleanup()

        # Return
        return train_losses_caption

# Azure Main Function
def main():
    param = {
        "azure": "True",
        "connection_string": "DefaultEndpointsProtocol=https;AccountName=shadetrainingdata;AccountKey=ARJHpGpb4tDyWY8MOc/4jtbbnbH+LF3W0r1pe9mReXbP+ZvdFWfMTwtKDowflQt8F2SgziL2gJ2Q+ASto1pcnQ==;EndpointSuffix=core.windows.net",
        "container_name": "dlip-dataset",
        "multi": "True",
        "ddp_server": "nccl",
        "queue_size": 57600,
        "image_size": 224,
        "batch_size": 160,
        "world_size": 4,
        "partial": 0,
        "val_split": 0.1,
        "device": "cuda",
        "scheduler": "True",
        "warmup_steps": 3000,
        "warmup_lr": 1e-6,
        "min_lr": 1e-6,
        "lr_decay_rate": 0.9,
        "dlip_bert": "tiny",
        "dlip_vit": "tiny",
        "accumulate": 0,
        "learning_rate": 5e-4,
        "weight_decay": 0.02,
        "dlip_epoch": 0,
        "checkpoint_epoch": 0,
        "num_epoch": 20,
        "freeze": "False",
        "gradient_clip": "False",
    }

    # DLIPCaptionTrain
    dlip_caption_train = DLIPCaptionTrain(azure=param['azure'],
                                           connection_string=param['connection_string'],
                                           container_name=param['container_name'],
                                           multi=param['multi'],
                                           ddp_server=param['ddp_server'],
                                           image_size=param['image_size'],
                                           batch_size=param['batch_size'],
                                           world_size=param['world_size'],
                                           partial=param['partial'],
                                           val_split=param['val_split'],
                                           device=param['device'],
                                           scheduler=param['scheduler'],
                                           warmup_steps=param['warmup_steps'],
                                           warmup_lr=param['warmup_lr'],
                                           min_lr=param['min_lr'],
                                           lr_decay_rate=param['lr_decay_rate'],
                                           dlip_bert=param['dlip_bert'],
                                           dlip_vit=param['dlip_vit'],
                                           accumulate=param['accumulate'],
                                           learning_rate=param['learning_rate'],
                                           weight_decay=param['weight_decay'],
                                           dlip_epoch=param['dlip_epoch'],
                                           checkpoint_epoch=param['checkpoint_epoch'],
                                           num_epoch=param['num_epoch'],
                                           freeze=param['freeze'],
                                           gradient_clip=param['gradient_clip'],
                                           )

    # Train
    if param['multi'] == "True":
        mp.spawn(
            dlip_caption_train.train,
            args=(),
            nprocs=dlip_caption_train.world_size
        )
    else:
        dlip_caption_train.train(0)

if __name__ == "__main__":
    main()
