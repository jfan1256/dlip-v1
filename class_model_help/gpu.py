import torch

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------MULTI-GPU PROCESSING------------------------------------------------------------------------------
# Concat operations across for Multi-GPU/Single-GPU operation
@torch.no_grad()
def concat_all_gather(tensor, multi):
    if multi:
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor

# Gather Layers for Multi-GPU/Single-GPU operation
class GatherLayer(torch.autograd.Function):
    multi = True

    @staticmethod
    def forward(ctx, x):
        if GatherLayer.multi:
            output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(output, x)
            return tuple(output)
        else:
            return x

    @staticmethod
    def backward(ctx, *grads):
        if GatherLayer.multi:
            all_gradients = torch.stack(grads)
            torch.distributed.all_reduce(all_gradients)
            return all_gradients[torch.distributed.get_rank()]
        else:
            return grads[0]

# Execute Concat and Gather Layers for Multi-GPU/Single-GPU operation
def all_gather_with_grad(tensor):
    if GatherLayer.multi:
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return tensor
        tensor_all = GatherLayer.apply(tensor)
        return torch.cat(tensor_all, dim=0)
    else:
        return tensor