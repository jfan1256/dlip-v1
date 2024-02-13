import io
import torch
import json
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from azure.storage.blob import BlobServiceClient

from class_data.azure_blob import AzureBlob
from class_train_help.metric_log import SmoothedValue, MetricLogger
from class_transform.randaugment import RandomAugment
from class_model.dlip import DLIP
from class_data.preprocess import Preprocess
from class_data.image_tensor import ImageTensor
from class_train_help.ddp_handle import DDPHandle
from class_train_help.metric_log import warmup_lr_schedule, step_lr_schedule
from utils.system import *

class DLIPTrain:
    def __init__(self,
                 azure=None,
                 connection_string=None,
                 container_name=None,
                 multi=None,
                 ddp_server=None,
                 queue_size=None,
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
                 dlip_bert_pretrain=None,
                 dlip_bert=None,
                 dlip_vit=None,
                 dlip_blip=None,
                 alpha=None,
                 accumulate=None,
                 learning_rate=None,
                 weight_decay=None,
                 checkpoint_epoch=None,
                 num_epoch=None,
                 freeze=None,
                 gradient_clip=None,
                 print=None,
                 ):

        '''
        azure (bool_str): Azure Cloud training or not
        connection_string (str): Connection string for Azure Blob Storage
        container_name (str): Name of the container in Azure Blob Storage
        multi (bool_str): Multi-GPU training or not
        ddp_server (str): Pytorch DDP Server (i.e., 'nccl' for Linux, 'gloo' for Windows)
        queue_size (int): Queue Size (must be a multiple of batch size)
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
        dlip_bert_pretrain (bool_str): Pretrain BERT or train BERT from scratch
        dlip_bert_config (dict): BERT config (only need if training BERT from scratch)
        dlip_bert (str): Size of BERT model (either 'tiny', 'base', or 'large')
        dlip_vit (str): Size of Vit model (either 'tiny', 'base', or 'large')
        dlip_blip (str): Str of BLIP model (either 'base' or 'large')
        alpha (float): Dlip alpha (default to 0.8)
        accumulate (int): Number of batches to accumulate before optimizer.step()
        learning_rate (float): Learning rate
        weight_decay (float): Learning rate decay
        checkpoint_epoch (int): Checkpoint epoch number for continuous training (i.e., continue training after model stopped training)
        num_epoch (int): Number of epochs to train
        freeze (bool_str): Freeze weights for training or not (to test whether the model is messing up or not)
        gradient_clip (bool_str): Gradient clipping or not (helpful for ita loss)
        print (bool_str): Print BLIP Captions or not
        '''

        self.azure = azure
        self.connection_string = connection_string
        self.container_name = container_name
        self.ddp_server = ddp_server
        self.multi = multi
        self.queue_size = queue_size
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
        self.dlip_bert_pretrain = dlip_bert_pretrain
        self.dlip_bert = dlip_bert
        self.dlip_vit = dlip_vit
        self.dlip_blip = dlip_blip
        self.alpha = alpha
        self.accumulate = accumulate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoint_epoch = checkpoint_epoch
        self.num_epoch = num_epoch
        self.freeze = freeze
        self.gradient_clip = gradient_clip
        self.print = print

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
            dataloader_train = multi_gpu.prepare(dataset_train, rank, self.world_size, self.batch_size, shuffle=True)
        else:
            dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
            dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, drop_last=True)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------CREATE MODEL-----------------------------------------------------------------------------------
        print("-"*60 + "\nCreating model...")
        # Create model
        if self.multi == "True":
            dlip = DLIP(image_size=self.image_size, queue_size=self.queue_size, bert_pretrain=self.dlip_bert_pretrain, bert=self.dlip_bert, vit=self.dlip_vit, blip=self.dlip_blip, print=self.print, multi=True).to(torch.device(f'cuda:{rank}'))
            dlip = DDP(dlip, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        else:
            dlip = DLIP(image_size=self.image_size, queue_size=self.queue_size, bert_pretrain=self.dlip_bert_pretrain, bert=self.dlip_bert, vit=self.dlip_vit, blip=self.dlip_blip, print=self.print, multi=False).to(torch.device(self.device))

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------CREATE OPTIMIZER/CHECKPOINT--------------------------------------------------------------------------
        print("-"*60 + "\nCreating optimizer...")
        # Create optimizer
        optimizer = optim.Adam(dlip.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Handle checkpoint
        if self.azure == "True":
            if self.checkpoint_epoch > 0:
                # Create a blob client using the local file name as the name for the blob
                blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                container_client = blob_service_client.get_container_client(self.container_name)

                checkpoint_blob_name = f'dlip/epoch_{self.checkpoint_epoch + 1}.pt'
                blob_client = container_client.get_blob_client(blob=checkpoint_blob_name)

                # Download the blob content to a stream
                checkpoint_stream = blob_client.download_blob().readall()

                # Load checkpoint from the stream
                checkpoint = torch.load(io.BytesIO(checkpoint_stream))
                dlip.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.alpha = checkpoint['alpha']
        else:
            if self.checkpoint_epoch > 0:
                # Load checkpoint model and optimizer
                checkpoint_path = get_data() / 'dlip' / f'epoch_{self.checkpoint_epoch + 1}.pt'
                checkpoint = torch.load(checkpoint_path)
                dlip.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                self.alpha = checkpoint['alpha']

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------START TRAINING---------------------------------------------------------------------------------
        print("-"*60 + "\nStarting train...")
        # Store losses
        train_losses_itm = []
        train_losses_ita = []
        train_losses_caption = []
        train_losses_dist = []
        val_losses_itm = []
        val_losses_ita = []
        val_losses_caption = []
        val_losses_dist = []

        # Store alpha
        alpha = self.alpha

        # Create Metric Loggers
        train_metric_logger = MetricLogger(delimiter="  ")
        train_metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        train_metric_logger.add_meter('train_loss_caption', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        train_metric_logger.add_meter('train_loss_itm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        train_metric_logger.add_meter('train_loss_ita', SmoothedValue(window_size=1, fmt='{value:.4f}'))
        train_metric_logger.add_meter('train_loss_dist', SmoothedValue(window_size=1, fmt='{value:.4f}'))

        # Iterate through num_epoch
        for epoch in range(self.checkpoint_epoch, self.num_epoch):
            # Train
            dlip.train()

            # Step linear scheduler
            if self.scheduler == "True":
                step_lr_schedule(optimizer, epoch, self.learning_rate, self.min_lr, self.lr_decay_rate)

            # Reshuffle dataset
            if self.azure == "True":
                dataloader_train.sampler.set_epoch(epoch)

            # Freeze All Weights
            if self.freeze == "True":
                for param in dlip.parameters():
                    param.requires_grad = False

            # Print Epoch
            header = 'Train Epoch: [{}]'.format(epoch+1)
            print_freq = 5

            # Initialize or Reset Losses for Each Epoch
            total_loss_ita, total_loss_itm, total_loss_caption, total_loss_dist = 0, 0, 0, 0
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
                    image = image.to(torch.device(f'cuda:{rank}'), non_blocking=True)
                    idx = idx.to(torch.device(f'cuda:{rank}'), non_blocking=True)
                else:
                    image = image.to(torch.device(self.device), non_blocking=True)
                    idx = idx.to(torch.device(self.device), non_blocking=True)

                # Increment Batch Count
                num_batches += 1

                # Set Trainable Alpha
                alpha = self.alpha * min(1, (epoch * len(dataloader_train) + i) / (2 * len(dataloader_train)))

                # Retrieve Losses from Dlip Class
                train_loss_ita, train_loss_itm, train_loss_lm, train_loss_dist = dlip(image=image, caption=caption, alpha=alpha, idx=idx)

                # Perform Optimization
                if self.gradient_clip == "True":
                    # Gradient Clipping
                    train_loss_ita = train_loss_ita
                    loss_other = train_loss_lm + train_loss_itm + train_loss_dist

                    # Average across accumulation
                    if self.accumulate != 0:
                        train_loss_ita /= self.accumulate
                        loss_other /= self.accumulate

                    if self.freeze != "True":
                        train_loss_ita.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(dlip.parameters(), 0.02)
                        loss_other.backward()

                    # Accumulate gradient
                    if self.accumulate != 0 and num_batches%self.accumulate == 0 and self.freeze != "True":
                        print(f"Accumulating gradient step: num_batches = {num_batches}...")
                        optimizer.step()
                    elif self.accumulate == 0:
                        optimizer.step()

                else:
                    loss_all = train_loss_ita + train_loss_itm + train_loss_lm + train_loss_dist
                    # Average across accumulation
                    if self.accumulate != 0:
                        loss_all /= self.accumulate

                    if self.freeze != "True":
                        loss_all.backward()

                    # Accumulate gradient
                    if self.accumulate != 0 and num_batches%self.accumulate == 0 and self.freeze != "True":
                        print(f"Accumulating gradient step: num_batches = {num_batches}...")
                        optimizer.step()
                    elif self.accumulate == 0:
                        optimizer.step()

                # Store Losses
                train_losses_itm.append(train_loss_itm.item())
                train_losses_ita.append(train_loss_ita.item())
                train_losses_caption.append(train_loss_lm.item())
                train_losses_dist.append(train_loss_dist.item())

                # Accumulate Losses
                total_loss_ita += train_loss_ita.item()
                total_loss_itm += train_loss_itm.item()
                total_loss_caption += train_loss_lm.item()
                total_loss_dist += train_loss_dist.item()

                # Log Losses
                train_metric_logger.update(train_loss_itm=train_loss_itm.item())
                train_metric_logger.update(train_loss_ita=train_loss_ita.item())
                train_metric_logger.update(train_loss_caption = train_loss_lm.item())
                train_metric_logger.update(train_loss_dist= train_loss_dist.item())
                train_metric_logger.update(lr=optimizer.param_groups[0]["lr"])

                # Get Validation Losses (only works for single gpu training)
                if (num_batches % 50 == 0 or num_batches == 1) and self.multi == "False" and self.val_split > 0:
                    # Set model to evaluation
                    print("-" * 60)
                    dlip.eval()

                    with torch.no_grad():
                        for i, (image, caption, idx) in enumerate(dataloader_val):
                            # Put data to device
                            image = image.to(torch.device(self.device))
                            idx = idx.to(torch.device(self.device))

                            # Get validation losses
                            val_loss_ita, val_loss_itm, val_loss_lm, val_loss_dist, val_loss_attn = dlip(image, caption, alpha=alpha, idx=idx)

                            # Log Validation Losses
                            val_losses_itm.append(val_loss_itm.item())
                            val_losses_ita.append(val_loss_ita.item())
                            val_losses_caption.append(val_loss_lm.item())
                            val_losses_dist.append(val_loss_dist.item())

                            if i % 10 == 0:
                                print(f"Validation [{i}/{len(dataloader_val)}]: val_loss_caption: {round(val_loss_lm.item(), 4)}, val_loss_itm: {round(val_loss_itm.item(), 4)}, val_loss_ita: {round(val_loss_ita.item(), 4)}, val_loss_dist: {round(val_loss_dist.item(), 4)}")

                    # Set model to train
                    print("-" * 60)
                    dlip.train()

            # Calculate Average Losses for the Epoch
            avg_loss_ita = total_loss_ita / num_batches
            avg_loss_itm = total_loss_itm / num_batches
            avg_loss_caption = total_loss_caption / num_batches
            avg_loss_dist = total_loss_dist / num_batches

            # Save Model Checkpoint
            if self.multi == "True":
                checkpoint = {
                    'epoch': epoch + 1,
                    'alpha': alpha,
                    'state_dict': dlip.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss_ita': avg_loss_ita,
                    'avg_loss_itm': avg_loss_itm,
                    'avg_loss_caption': avg_loss_caption,
                    'avg_loss_dist': avg_loss_dist,
                }
            else:
                checkpoint = {
                    'epoch': epoch + 1,
                    'alpha': alpha,
                    'state_dict': dlip.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss_ita': avg_loss_ita,
                    'avg_loss_itm': avg_loss_itm,
                    'avg_loss_caption': avg_loss_caption,
                    'avg_loss_dist': avg_loss_dist,
                }

            if self.azure == "True":
                # Create a blob client
                blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                container_client = blob_service_client.get_container_client(self.container_name)

                # Name of the blob (file in Azure)
                blob_name = f'dlip/epoch_{epoch + 1}.pt'

                # Create a BlobClient
                blob_client = container_client.get_blob_client(blob=blob_name)

                # Convert the checkpoint dictionary to a byte stream
                checkpoint_stream = io.BytesIO()
                torch.save(checkpoint, checkpoint_stream)
                checkpoint_stream.seek(0)

                # Upload the stream to Azure Blob Storage
                blob_client.upload_blob(checkpoint_stream, overwrite=True)
            else:
                checkpoint_path = get_data() / 'dlip' / f'epoch_{epoch + 1}.pt'
                os.makedirs(checkpoint_path.parent, exist_ok=True)
                torch.save(checkpoint, checkpoint_path)

            # Save Losses
            losses_dict = {
                "train_losses_itm": train_losses_itm,
                "train_losses_ita": train_losses_ita,
                "train_losses_caption": train_losses_caption,
                "train_losses_dist": train_losses_dist,
                "val_losses_itm": val_losses_itm,
                "val_losses_ita": val_losses_ita,
                "val_losses_caption": val_losses_caption,
                "val_losses_dist": val_losses_dist
            }

            if self.azure == "True":
                json_string = json.dumps(losses_dict)
                bytes_stream = io.BytesIO(json_string.encode())

                # Name of the blob (file in Azure)
                blob_name = f'log/losses.json'

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
                log_file_path = get_data() / 'log' / 'losses.json'
                with open(log_file_path, 'w') as file:
                    json.dump(losses_dict, file)

            if self.azure == "True":
                dist.barrier()

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------------CLEAN UP------------------------------------------------------------------------------------
        # Clean up multi-gpu process
        if self.multi == "True":
            multi_gpu.cleanup()

        # Return
        return train_losses_itm, train_losses_ita, train_losses_caption, train_losses_dist, val_losses_itm, val_losses_ita, val_losses_caption, val_losses_dist

# Azure Main Function
def main():
    param = {
      "azure" : "True",
      "connection_string" : "",
      "container_name" : "dlip-dataset",
      "multi" : "True",
      "ddp_server": "nccl",
      "queue_size" : 57600,
      "image_size" : 224,
      "batch_size" : 160,
      "world_size" : 4,
      "partial" : 0,
      "val_split" : 0.1,
      "device" : "cuda",
      "scheduler" : "True",
      "warmup_steps" : 3000,
      "warmup_lr" : 1e-6,
      "min_lr" : 1e-6,
      "lr_decay_rate" : 0.9,
      "dlip_bert_pretrain": "False",
      "dlip_bert" : "tiny",
      "dlip_vit" : "tiny",
      "dlip_blip" : "base",
      "alpha" : 0.4,
      "accumulate" : 0,
      "learning_rate" : 5e-4,
      "weight_decay" : 0.02,
      "checkpoint_epoch" : 0,
      "num_epoch" : 20,
      "freeze" : "False",
      "gradient_clip" : "False",
      "print" : "False"
    }

    # DLIPTrain
    dlip_train = DLIPTrain(azure=param['azure'],
                           connection_string=param['connection_string'],
                           container_name=param['container_name'],
                           multi=param['multi'],
                           ddp_server=param['ddp_server'],
                           queue_size=param['queue_size'],
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
                           dlip_bert_pretrain=param['dlip_bert_pretrain'],
                           dlip_bert=param['dlip_bert'],
                           dlip_vit=param['dlip_vit'],
                           dlip_blip=param['dlip_blip'],
                           alpha=param['alpha'],
                           accumulate=param['accumulate'],
                           learning_rate=param['learning_rate'],
                           weight_decay=param['weight_decay'],
                           checkpoint_epoch=param['checkpoint_epoch'],
                           num_epoch=param['num_epoch'],
                           freeze=param['freeze'],
                           gradient_clip=param['gradient_clip'],
                           print=param['print']
                           )

    # Train
    if param['multi'] == "True":
        mp.spawn(
            dlip_train.train,
            args=(),
            nprocs=dlip_train.world_size
        )
    else:
        dlip_train.train(0)

if __name__ == "__main__":
    main()
