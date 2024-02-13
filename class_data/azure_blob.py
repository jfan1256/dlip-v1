import re
import io
import pandas as pd

from azure.storage.blob import BlobServiceClient, ContainerClient

class AzureBlob:
    def __init__(self,
                 connection_string=None,
                 container_name=None,
                 file_pattern=None
                 ):

        '''
        connection_string (str): Connection string for Azure Blob Storage
        container_name (str): Name of the container in Azure Blob Storage
        file_pattern (str): File pattern to concat similar files
        '''

        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        self.file_pattern = file_pattern

    # Concat files with same file pattern name in Azure Blob Storage Account
    def _concat_files(self):
        def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
            return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

        # List all blobs in the container and match the pattern
        blob_list = self.container_client.list_blobs()
        matching_blobs = []
        for blob in blob_list:
            if re.search(self.file_pattern, blob.name):
                matching_blobs.append(blob)

            if len(matching_blobs) == 10:
                break

        # Sort blobs using natural sort
        matching_blobs.sort(key=lambda blob: natural_sort_key(blob.name))

        df_list = []
        for blob in matching_blobs:
            print(f"Found matching blob: {blob.name}")
            blob_client = self.container_client.get_blob_client(blob=blob.name)
            stream = blob_client.download_blob().readall()
            df = pd.read_parquet(io.BytesIO(stream))
            df_list.append(df)

        data = pd.concat(df_list, axis=0)
        return data
