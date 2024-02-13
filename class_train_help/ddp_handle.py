import os
import torch.distributed as dist
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

class DDPHandle:
    @staticmethod
    # Set up the process group
    def setup(rank, world_size, ddp_server):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(ddp_server, rank=rank, world_size=world_size)

    @staticmethod
    # Split the dataloader
    def prepare(dataset, rank, world_size, batch_size, shuffle):
        # Set pin_memory to False and num_workers to 0 to avoid issues
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, drop_last=True, sampler=sampler)
        return dataloader

    @staticmethod
    def cleanup():
        dist.destroy_process_group()