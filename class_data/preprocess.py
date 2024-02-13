import re
import ray
import numpy as np
import glob
import pandas as pd

from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class Preprocess:
    def __init__(self,
                 folder_path=None,
                 file_name=None,
                 file_pattern=None,
                 data=None,
                 column_name=None,
                 default_name=None,
                 type=None,
                 max_words=None
                 ):

        '''
        folder_path (str): Folder path to retrieve files
        file_name (str): Filename for export
        file_pattern (str): File pattern to concat similar files
        data (pd.DataFrame): Dataframe to be preprocessed
        column_name (string): Column name for new preprocessed column
        default_name (string): Column name for column to be preprocessed
        type (string): Name of dataset (i.e., coco)
        max_words (int): Max amount of words per caption
        '''

        self.folder_path = folder_path
        self.file_name = file_name
        self.file_pattern = file_pattern
        self.data = data
        self.column_name = column_name
        self.default_name = default_name
        self.type = type
        self.max_words = max_words


    # Export large dataframe into parquet chunks
    def _export_in_chunks(self):
        chunks = np.array_split(self.data, 10)
        for i, df in enumerate(chunks, 1):
            print(f"Exporting chunk: {i}/{len(chunks)}")
            df.to_parquet(self.folder_path / f'{self.file_name}_{i}.parquet.brotli', compression='brotli')

    # Concat files with same file pattern name
    def _concat_files(self):
        def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

        full_pattern = f'{self.folder_path}/{self.file_pattern}'
        file_list = glob.glob(full_pattern)
        file_list.sort(key=natural_sort_key)
        df_list = []
        for file in file_list:
            df = pd.read_parquet(file)
            df_list.append(df)
        data = pd.concat(df_list, axis=0)
        return data

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

    # Preprocess Text
    def _preprocess(self):
        self.data[self.column_name] = self.data[self.column_name].astype(str).apply(self.pre_caption)
        self.data.index.names = ['id']
        self.data['type'] = self.type
        return self.data

    # Parallelized Caption Tokenization
    def _tokenize_caption(self, num_cpu):
        ray.init(num_cpus=num_cpu, ignore_reinit_error=True)

        # Load BERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        @ray.remote
        def caption_to_tokenize(caption):
            return tokenizer(caption, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)

        # Determine batch size
        batch_size = max(1, len(self.data[self.default_name]) // (num_cpu * num_cpu))
        all_tokenized_captions = []

        # Process each batch sequentially
        total_batches = (len(self.data[self.default_name]) + batch_size - 1) // batch_size
        for i in range(0, len(self.data[self.default_name]), batch_size):
            current_batch = i // batch_size + 1
            print(f"Processing batch: {current_batch}/{total_batches}")
            batch = self.data[self.default_name][i:i + batch_size]
            futures = [caption_to_tokenize.remote(caption) for caption in batch]
            results = ray.get(futures)
            for result in results:
                all_tokenized_captions.extend(result["input_ids"].tolist())

        self.data[self.column_name] = all_tokenized_captions

        ray.shutdown()
        return self.data

    # Parallelized Image Vectorization
    def _vectorize_image(self, transform, num_cpu):
        ray.init(num_cpus=num_cpu, ignore_reinit_error=True)

        @ray.remote
        def image_to_tensor(file_path, transform):
            try:
                with Image.open(file_path) as img:
                    if transform:
                        img = transform(img)
                    else:
                        img = transforms.ToTensor()(img)
                    return img
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
                return None

        # Determine batch size
        batch_size = max(1, len(self.data[self.default_name]) // (num_cpu * num_cpu))

        # Initialize variables for results
        all_image_tensor = []

        # Process each batch sequentially
        total_batches = (len(self.data[self.default_name]) + batch_size - 1) // batch_size
        for i in range(0, len(self.data[self.default_name]), batch_size):
            current_batch = i // batch_size + 1
            print(f"Processing batch: {current_batch}/{total_batches}")
            batch = self.data[self.default_name][i:i + batch_size]
            futures = [image_to_tensor.remote(self.folder_path / f'{img}', transform) for img in batch]
            result = ray.get(futures)
            all_image_tensor.extend(result)

        self.data.loc[:, self.column_name] = all_image_tensor
        ray.shutdown()
        return self.data