import json
import time

import numpy as np
import torch
import pandas as pd

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from class_model.dlip import DLIP
from utils.system import get_config
from class_model_help.pretrain import init_blip_caption
from utils.system import get_data
from class_data.preprocess import Preprocess
from class_data.image_tensor import ImageTensor

# Evaluate DLIP Base Model
class DLIPEval:
    def __init__(self,
                 device=None,
                 epoch_number=None,
                 dlip_bert_pretrain=None,
                 dlip_bert=None,
                 dlip_vit=None,
                 dlip_blip=None,
                 partial=None,
                 val_split=None,
                 print=None
                 ):

        '''
        device (str): Device to evaluate on
        epoch_number (str): Epoch number of the model checkpoint to use
        dlip_bert_pretrain (bool_str): Pretrain BERT or train BERT from scratch
        dlip_bert (str): Size of BERT model (this should be the same size as the trained model)
        dlip_vit (str): Size of Vit model (this should be the same size as the trained model)
        dlip_blip (str): Size of BLIP model (either 'base' or 'large')
        partial (bool_str): Evaluate on partial data or not
        val_split (float): Fraction of data to allocate for validation
        print (bool_str): Generate BLIP captions or not
        '''

        self.device = torch.device(device)
        self.epoch_number = epoch_number
        self.dlip_bert_pretrain = dlip_bert_pretrain
        self.dlip_bert = dlip_bert
        self.dlip_vit = dlip_vit
        self.dlip_blip = dlip_blip
        self.partial = partial
        self.val_split = val_split
        self.print = print

    def evaluate(self):
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------LOAD DATA--------------------------------------------------------------------------------------
        print("-"*60 + "\nLoading data...")
        # Load Data
        file_pattern = 'all_data_tokenize_*.parquet.brotli'
        folder_path = get_data() / 'all' / 'chunks'
        all_data = Preprocess(folder_path=folder_path, file_pattern=file_pattern)._concat_files()
        val_data = all_data.sample(frac=self.val_split, random_state=42)

        # Train on partial data
        if self.partial> 0:
            all_data = all_data.sample(n=self.partial, random_state=42)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------CREATE DATALOADER-----------------------------------------------------------------------------------
        print("-"*60 + "\nCreating dataloader...")
        # Create Transformation
        image_size = 224
        batch_size = 16

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        transform_eval = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])

        dataset_generate = ImageTensor(data=all_data, image_column='image_name', caption_column='caption', transform=transform_eval)
        dataset_val = ImageTensor(data=val_data, image_column='image_name', caption_column='caption', transform=transform_eval)
        dataloader_generate = DataLoader(dataset_generate, batch_size=batch_size, shuffle=False, drop_last=False)
        dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, drop_last=True)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------LOAD MODEL-------------------------------------------------------------------------------------
        print("-"*60 + "\nLoading model...")
        # Load checkpoint
        checkpoint_path = get_data() / 'dlip'
        checkpoint_files = [f for f in checkpoint_path.glob('*.pt')]

        checkpoints = {}
        for file in checkpoint_files:
            epoch_number = file.stem.split('_')[-1]
            if epoch_number == self.epoch_number:
                checkpoints[epoch_number] = torch.load(file, map_location=self.device)

        dlip = DLIP(image_size=image_size, bert_pretrain=self.dlip_bert_pretrain, bert=self.dlip_bert, vit=self.dlip_vit, blip=self.dlip_blip, multi=False).to(self.device)
        state_dict = checkpoints[self.epoch_number]['state_dict']

        # Filter blip_caption and blip_retrieval weights out
        if self.dlip_blip != None:
            dlip.load_state_dict(state_dict)
        else:
            state_dict = {name: param for name, param in state_dict.items() if not any(ex_layer in name for ex_layer in ['blip_caption', 'blip_retrieval'])}
            dlip.load_state_dict(state_dict)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------EVALUATE MODEL------------------------------------------------------------------------------------
        val_losses_itm = []
        val_losses_ita = []
        val_losses_caption = []
        val_losses_hr = []
        val_losses_attn = []

        if self.val_split != 0:
            # Set model to evaluation
            print("-"*60 + "\nEvaluating model...\n" + "-"*60)
            dlip.eval()

            with torch.no_grad():
                # Freeze weights
                for param in dlip.parameters():
                    param.requires_grad = False

                for i, (image, caption, idx) in enumerate(dataloader_val):
                    # Put data to device
                    image = image.to(torch.device(self.device))
                    idx = idx.to(torch.device(self.device))

                    # Get validation losses
                    val_loss_ita, val_loss_itm, val_loss_lm, val_loss_hr, val_loss_attn = dlip(image, caption, alpha=0.8, idx=idx)

                    # Log Validation Losses
                    val_losses_itm.append(val_loss_itm.item())
                    val_losses_ita.append(val_loss_ita.item())
                    val_losses_caption.append(val_loss_lm.item())
                    val_losses_hr.append(val_loss_hr.item())
                    val_losses_attn.append(val_loss_attn.item())

                    if i % 10 == 0:
                        print(f"Validation [{i}/{len(dataloader_val)}]: val_loss_caption: {round(val_loss_lm.item(), 4)}, val_loss_itm: {round(val_loss_itm.item(), 4)}, val_loss_ita: {round(val_loss_ita.item(), 4)}, val_loss_hr: {round(val_loss_hr.item(), 4)}, val_loss_attn: {round(val_loss_attn.item(), 4)}")

            print("-" * 60)
            print(f"Average Validation: avg_val_loss_lm: {round(np.array(val_losses_caption).mean(), 4)}, avg_val_loss_itm: {round(np.array(val_losses_itm).mean(), 4)}, avg_val_loss_ita: {round(np.array(val_losses_ita).mean(), 4)}, avg_val_loss_hr: {round(np.array(val_losses_hr).mean(), 4)}, avg_val_loss_attn: {round(np.array(val_losses_attn).mean(), 4)}")
            # Set model to train
            print("-" * 60)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------GENERATE CAPTION----------------------------------------------------------------------------------
        print("-"*60 + "\nGenerating caption...\n" + "-"*60)
        # Evaluate model
        all_caption = []

        # Initialize BLIP
        if self.print == "True":
            self.process, self.caption, _ = init_blip_caption('base')
            self.caption = self.caption.to(self.device)

        # Processing images in the dataloader (set sample=True for generate() to work)
        start_time = time.time()
        for (image, caption, idx) in dataloader_generate:
            # Generate DLIP caption
            image_time = time.time()
            image = image.to(self.device)
            dlip_caption = dlip.generate(image)
            generate_time = time.time() - image_time
            image_per_ms = generate_time * 1000
            print(f"Image per second with batch size = {batch_size}: {image_per_ms:.2f} milliseconds")

            # Generate BLIP caption
            if self.print == "True":
                # Create compatible image for parent model
                img_min = image.min()
                img_max = image.max()
                parent_image = (image - img_min) / (img_max - img_min)
                parent_image.mul_(255).clamp_(0, 255)
                parent_image = parent_image.to(image.device, dtype=torch.uint8)

                # Generate caption
                blip_input = self.process(images=parent_image, return_tensors="pt").to(image.device)
                with torch.no_grad():
                    blip_captions = self.caption.generate(**blip_input)
                blip_caption = [self.process.decode(caption, skip_special_tokens=True) for caption in blip_captions]

                for id, dlip_cap, org_cap, blip_cap in zip(idx, dlip_caption, caption, blip_caption):
                    all_caption.append([id.item(), dlip_cap, org_cap, blip_cap])
            else:
                for id, dlip_cap, org_cap in zip(idx, dlip_caption, caption):
                    all_caption.append([id.item(), dlip_cap, org_cap])

        # Calculate and display speed
        elasped_time = time.time() - start_time
        image_per_ms = (elasped_time/len(all_data))*1000
        print(f"Total time to produce caption for {len(all_data)} images: {elasped_time} seconds")
        print(f"Average time to produce caption per image: {image_per_ms:.2f} milliseconds")

        # Create a DataFrame
        if self.print == "True":
            evaluate = pd.DataFrame(all_caption, columns=['id', 'dlip_caption', 'original_caption', 'blip_caption'])
            evaluate = evaluate.set_index('id').sort_index()
        else:
            evaluate = pd.DataFrame(all_caption, columns=['id', 'dlip_caption', 'original_caption'])
            evaluate = evaluate.set_index('id').sort_index()

        return evaluate, val_losses_itm, val_losses_ita, val_losses_caption, val_losses_hr, val_losses_attn

# DlipEval Main Function
def main():
    # Get Device and Number of Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Params
    param = json.load(open(get_config() / 'train_config.json'))
    epoch_number = '2'
    dlip_bert_pretrain = param['dlip_bert_pretrain']
    dlip_bert = param['dlip_bert']
    dlip_vit = param['dlip_vit']
    dlip_blip = param['dlip_blip']
    partial = "True"
    print = "True"
    val_split = 0

    # DLIPEval
    dlip_eval = DLIPEval(device=device,
                         epoch_number=epoch_number,
                         dlip_bert_pretrain=dlip_bert_pretrain,
                         dlip_bert=dlip_bert,
                         dlip_vit=dlip_vit,
                         dlip_blip=dlip_blip,
                         partial=partial,
                         val_split=val_split,
                         print=print)

    # Evaluate
    evaluate, val_losses_itm, val_losses_ita, val_losses_caption, val_losses_hr, val_losses_attn = dlip_eval.evaluate()

if __name__ == "__main__":
    main()