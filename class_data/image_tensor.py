import io
import urllib
import re

from PIL import Image
from torch.utils.data import Dataset
from azure.storage.blob import BlobServiceClient
from datasets.utils.file_utils import get_datasets_user_agent

from class_data.azure_blob import AzureBlob
from class_data.preprocess import Preprocess
from utils.system import *

USER_AGENT = get_datasets_user_agent()

class ImageTensor(Dataset):
    def __init__(self,
                 azure=None,
                 connection_string=None,
                 container_name=None,
                 data=None,
                 image_column=None,
                 caption_column=None,
                 transform=None,
                 max_words=None
                 ):

        '''
        Data format is as follows:
            data.index.names = 'id'
            data[caption_column] = column of image captions
            data[image_column] = column of image names (i.e., 100203.png)

        azure (bool_str): Azure Blob storage or not
        connection_string (str): Connection string for Azure Blob Storage
        container_name (str): Name of the container in Azure Blob Storage
        data (pd.DataFrame): DataFrame containing image file names.
        image_column (str): Column name in DataFrame for image file names.
        caption_column (str): Column name in DataFrame for captions
        transform (torchvision.transforms): Transformations to be applied to each image.
        max_words (int): Maximum number of words per caption

        Returns image (actual image file), caption (length of text), idx (image ID in pandas dataframe)
        '''

        self.azure = azure
        self.data = data
        self.image_column = image_column
        self.caption_column = caption_column
        self.transform = transform
        self.connection_string = connection_string
        self.container_name = container_name
        self.max_words = max_words

        if self.azure == 'True':
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            self.container_client = self.blob_service_client.get_container_client(self.container_name)

        # Load Data
        if self.azure == "True":
            file_pattern = r'download_data_tokenize_\d+\.parquet\.brotli'
            self.download = AzureBlob(connection_string=self.connection_string, container_name=self.container_name, file_pattern=file_pattern)._concat_files()
        else:
            file_pattern = 'download_data_tokenize_*.parquet.brotli'
            folder_path = get_data() / 'download' / 'chunks'
            self.download = Preprocess(folder_path=folder_path, file_pattern=file_pattern)._concat_files()


    # Pre caption
    def pre_caption(self, caption):
        caption = re.sub(r"([.!\"()*#:;~])", ' ', caption.lower())
        caption = re.sub(r"\s{2,}", ' ', caption)
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # Truncate caption
        caption_words = caption.split(' ')
        if len(caption_words) > self.max_words:
            caption = ' '.join(caption_words[:self.max_words])

        return caption
    @staticmethod
    # Fetch image from url
    def fetch_image_url(img_url, timeout=None, retries=0):
        for _ in range(retries + 1):
            try:
                request = urllib.request.Request(
                    img_url,
                    data=None,
                    headers={"user-agent": USER_AGENT},
                )
                with urllib.request.urlopen(request, timeout=timeout) as req:
                    image = Image.open(io.BytesIO(req.read())).convert('RGB')
                break
            except Exception:
                image = None
        return image

    # Handle Image=None for URL Fetch
    def handle_none_url(self):
        random_row = self.download.sample(n=1)
        img_type = random_row.iloc[0]['type']
        img_name = random_row.iloc[0][self.image_column]
        caption = random_row.iloc[0][self.caption_column]
        idx = random_row.index[0]
        if self.azure == 'True':
            _, image, _ = self.fetch_image_azure(True, img_name, img_type, idx)
        else:
            _, image, _ = self.fetch_image_prem(True, img_name, img_type, idx)

        return idx, image, caption

    # Get Image for Azure
    def fetch_image_azure(self, download, img_name, img_type, idx):
        index, folder_path, caption = None, None, None
        if img_type == 'coco_train':
            folder_path = 'data/coco/images/train2017'
        elif img_type == 'coco_val':
            folder_path = 'data/coco/images/val2017'
        elif img_type == 'flickr30k_train':
            folder_path = 'data/flickr30k/images'
        image_blob = f"{folder_path}/{img_name}"
        blob_client = self.container_client.get_blob_client(blob=image_blob)
        stream = blob_client.download_blob().readall()
        image = Image.open(io.BytesIO(stream)).convert('RGB')

        if download == False:
            caption = self.data.iloc[idx][self.caption_column]
            index = self.data.index[idx]
        return index, image, caption

    # Get Image for On Prem
    def fetch_image_prem(self, download, img_name, img_type, idx):
        index, folder_path, caption = None, None, None
        if img_type == 'coco_train':
            folder_path = get_data() / 'coco' / 'images' / 'train2017'
        elif img_type == 'coco_val':
            folder_path = get_data() / 'coco' / 'images' / 'val2017'
        elif img_type == 'flickr30k_train':
            folder_path = get_data() / 'flickr30k' / 'images'
        img_name = os.path.join(folder_path, img_name)
        image = Image.open(img_name).convert('RGB')

        if download == False:
            caption = self.data.iloc[idx][self.caption_column]
            index = self.data.index[idx]
        return index, image, caption

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index, image, caption = None, None, None
        img_type = self.data.iloc[idx]['type']
        img_name = self.data.iloc[idx][self.image_column]

        if self.azure == 'True':
            # Azure Blob Storage Account
            if img_type in ['coco_train', 'coco_val', 'flickr30k_train']:
                index, image, caption = self.fetch_image_azure(False, img_name, img_type, idx)
            elif img_type in ['gcc_train', 'gcc_val', 'sbu_train']:
                caption = self.data.iloc[idx][self.caption_column]
                index = self.data.index[idx]
                image = self.fetch_image_url(img_name, timeout=30, retries=3)
                if image == None:
                    index, image, caption = self.handle_none_url()
        else:
            # Local On Prem Storage
            if img_type in ['coco_train', 'coco_val', 'flickr30k_train']:
                index, image, caption = self.fetch_image_prem(False, img_name, img_type, idx)
            elif img_type in ['gcc_train', 'gcc_val', 'sbu_train']:
                caption = self.data.iloc[idx][self.caption_column]
                index = self.data.index[idx]
                image = self.fetch_image_url(img_name, timeout=30, retries=3)
                if image == None:
                    index, image, caption = self.handle_none_url()

        if self.transform:
            image = self.transform(image)

        caption = self.pre_caption(caption)

        return image, caption, index