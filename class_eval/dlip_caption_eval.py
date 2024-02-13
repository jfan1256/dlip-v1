import time

import numpy as np
import torch
import pandas as pd

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader

from class_model.dlip import DLIP
from class_model.dlip_caption import DLIPCaption
from utils.system import get_data
from class_data.preprocess import Preprocess
from class_data.image_tensor import ImageTensor

# Evaluate DLIP Finetune Caption Model
class DLIPCaptionEval:
    def __init__(self,
                 device=None,
                 epoch_number=None,
                 dlip_bert=None,
                 dlip_vit=None,
                 dlip_blip=None,
                 partial=None,
                 val_split=None
                 ):

        '''
        device (str): Device to evaluate on
        epoch_number (str): Epoch number of the model checkpoint to use
        dlip_bert (str): Size of BERT model (this should be the same size as the trained model)
        dlip_vit (str): Size of Vit model (this should be the same size as the trained model)
        partial (bool_str): Evaluate on partial data or not
        val_split (float): Fraction of data to allocate for validation
        '''

        self.device = torch.device(device)
        self.epoch_number = epoch_number
        self.dlip_bert = dlip_bert
        self.dlip_vit = dlip_vit
        self.dlip_blip = dlip_blip
        self.partial = partial
        self.val_split = val_split

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
        if self.partial == "True":
            all_data = all_data.head(100)
            val_data = val_data.head(500)

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
            normalize,
        ])

        dataset_generate = ImageTensor(data=all_data, image_column='image_name', caption_column='caption', transform=transform_eval)
        dataset_val = ImageTensor(data=val_data, image_column='image_name', caption_column='caption', transform=transform_eval)
        dataloader_generate = DataLoader(dataset_generate, batch_size=batch_size, shuffle=False, drop_last=False)
        dataloader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, drop_last=True)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------LOAD MODEL-------------------------------------------------------------------------------------
        print("-" * 60 + "\nLoading model...")
        # Load checkpoint
        checkpoint_path = get_data() / 'dlip_caption'
        checkpoint_files = [f for f in checkpoint_path.glob('*.pt')]

        checkpoints = {}
        for file in checkpoint_files:
            epoch_number = file.stem.split('_')[-1]
            if epoch_number == self.epoch_number:
                checkpoints[epoch_number] = torch.load(file, map_location=self.device)

        dlip_caption = DLIPCaption(image_size=image_size, bert=self.dlip_bert, vit=self.dlip_vit, multi=False).to(self.device)
        state_dict = checkpoints[self.epoch_number]['state_dict']

        # Filter blip_caption and blip_retrieval weights out
        dlip_caption.load_state_dict(state_dict)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------EVALUATE MODEL------------------------------------------------------------------------------------
        val_losses_caption = []

        if self.val_split != 0:
            # Set model to evaluation
            print("-"*60 + "\nEvaluating model...\n" + "-"*60)
            dlip_caption.eval()

            with torch.no_grad():
                # Freeze weights
                for param in dlip_caption.parameters():
                    param.requires_grad = False

                for i, (image, caption, idx) in enumerate(dataloader_val):
                    # Put data to device
                    image = image.to(torch.device(self.device))
                    idx = idx.to(torch.device(self.device))

                    # Get validation losses
                    val_loss_lm = dlip_caption(image, caption)

                    # Log Validation Losses
                    val_losses_caption.append(val_loss_lm.item())

                    if i % 10 == 0:
                        print(f"Validation [{i}/{len(dataloader_val)}]: val_loss_caption: {round(val_loss_lm.item(), 4)}")

            print("-" * 60)
            print(f"Average Validation: avg_val_loss_lm: {round(np.array(val_losses_caption).mean(), 4)}")
            # Set model to train
            print("-" * 60)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------GENERATE CAPTION----------------------------------------------------------------------------------
        print("-"*60 + "\nGenerating caption...\n" + "-"*60)
        # Evaluate model
        all_caption = []

        # Processing images in the dataloader (set sample=True for generate() to work)
        start_time = time.time()
        for (image, caption, idx) in dataloader_generate:
            image_time = time.time()
            image = image.to(self.device)
            generated_caption = dlip_caption.generate(image, sample=True)
            generate_time = time.time() - image_time
            image_per_ms = generate_time * 1000
            print(f"Image per second with batch size = {batch_size}: {image_per_ms:.2f} milliseconds")
            for id, gen_cap, org_cap in zip(idx, generated_caption, caption):
                all_caption.append([id.item(), gen_cap, org_cap])

        # Calculate and display speed
        elasped_time = time.time() - start_time
        image_per_ms = (elasped_time/len(all_data))*1000
        print(f"Total time to produce caption for {len(all_data)} images: {elasped_time} seconds")
        print(f"Average time to produce caption per image: {image_per_ms:.2f} milliseconds")

        # Create a DataFrame
        evaluate = pd.DataFrame(all_caption, columns=['id', 'generate_caption', 'original_caption'])
        evaluate = evaluate.set_index('id').sort_index()

        return evaluate, val_losses_caption