import torch
import torch.nn.functional as F

from torch import nn
from torch.nn import MSELoss

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------DISTILLATION LOSS FUNCTIONS--------------------------------------------------------------------------------
# Aggregate num_layers for Hidden Representation
def aggregate_layers(tensor, target_layers):
    # Dimensions
    num_layers, batch_size, feature_dim, other_dim = tensor.shape

    # Return if target_layers is met
    if num_layers == target_layers:
        return tensor

    # Calculate the aggregation factor
    aggregation_factor = num_layers / target_layers
    new_tensor = []

    for i in range(target_layers):
        # Calculate the start and end indices of layers in the original tensor that will be aggregated to form a single layer in the new tensor
        start_idx = int(i * aggregation_factor)
        end_idx = int((i + 1) * aggregation_factor)
        # Aggregate (average) the layers from start_idx to end_idx and reduce them to a single layer
        aggregated_layer = tensor[start_idx:end_idx].mean(dim=0)
        new_tensor.append(aggregated_layer)

    return torch.stack(new_tensor)

# Hidden Representation Loss
def hr_loss(linear, parent_hr, student_hr, device):
    '''
    Note: parent_hr and student_hr should be 4D [num_layers, batch_size, num_states, embed_size]

    It is assumed that the parent_hr has more layers than student_hr
    and that parent_hr's sequence length is greater than student_attn's sequence length

    The current code outputs these dimensions:
        BaseBLIPCaption: parent_attn.shape = [25, 5, 577, 768]
        TinyVit: student_attn.shape = [13, 5, 197, 192]
        TinyBert: student_attn.shape = [3, 5, 35, 128]
    '''

    # Batch size
    batch_size = parent_hr.shape[1]

    # Aggregate Layers (whichever hr representation has more layers will be aggregated to have the same number of layers as the smaller hr representation)
    target_layers = min(parent_hr.size(0), student_hr.size(0))
    parent_aggregated = aggregate_layers(parent_hr, target_layers)
    student_aggregated = aggregate_layers(student_hr, target_layers)

    # Unstack parent_hr and student_hr
    parent_unstacked = torch.unbind(parent_aggregated, dim=0)
    student_unstacked = torch.unbind(student_aggregated, dim=0)

    # Setup Loss Function Variables
    loss_function = torch.nn.CosineEmbeddingLoss()
    ones = torch.ones(student_aggregated.size(1))
    total_loss = torch.tensor(0.0).to(device)
    num_elements = 0

    # HR Get Loss of Each Layer
    for parent, student in zip(parent_unstacked, student_unstacked):
        # Average across num_states
        parent_mean = parent.mean(1)
        student_mean = student.mean(1)

        # Project parent embeddings to student embeddings
        parent_proj = linear(parent_mean)

        # Calculate Loss
        current_loss = loss_function(parent_proj, student_mean, ones.to(device))

        # Average across batches
        current_loss /= batch_size

        # Accumulate loss
        total_loss += current_loss
        num_elements += 1

    # Average the HR Loss across N layers
    mean_loss = total_loss / num_elements
    return mean_loss

# Attention Loss
def attn_loss(parent_attn, student_attn):
    '''
    Note: parent_attn and student_attn should be 4D [batch_size, num_attn_heads, seq_length, seq_length]

    It is assumed that the parent_attn has more attention heads than student_attn
    and that parent_attn's sequence length is greater than student_attn's sequence length

    The current code outputs these dimensions:
        BaseBLIPCaption: parent_attn.shape = [5, 16, 577, 577]
        TinyVit: student_attn.shape = [5, 3, 197, 197]
        TinyBert: student_attn.shape = [5, 2, 35, 35]
    '''

    # Initialize Loss Function
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean')

    # Calculate the head sampling rate
    parent_heads = parent_attn.shape[1]
    student_heads = student_attn.shape[1]
    head_sample_rate = int(parent_heads / student_heads)

    # Sample heads from the parent attention using advanced indexing
    indices = torch.arange(0, parent_heads, head_sample_rate)[:student_heads]
    sampled_parent_attn = parent_attn[:, indices, :, :]

    # Use adaptive_avg_pool3d for pooling across the spatial dimensions (last two dimensions)
    resized_parent_attn = F.adaptive_avg_pool3d(sampled_parent_attn.unsqueeze(2),(student_heads, student_attn.shape[2], student_attn.shape[3])).squeeze(2)
    resized_student_attn = F.adaptive_avg_pool3d(student_attn.unsqueeze(2),(student_heads, student_attn.shape[2], student_attn.shape[3])).squeeze(2)

    # Calculate loss (average across batches)
    loss = kl_loss(F.log_softmax(resized_student_attn, dim=-1), F.softmax(resized_parent_attn, dim=-1))

    return loss
