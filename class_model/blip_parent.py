import torch

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers import BlipForImageTextRetrieval, BlipForConditionalGeneration
import torch.nn.functional as F

# BlipParent (outputs hidden representation and attention weights for each model for distillation loss)
class BlipParent:
    def __init__(self):
        # Load BLIP Caption Model
        # captioning = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
        # self.image_model_captioning = captioning.vision_model
        # self.text_decoder = captioning.text_decoder
        # del captioning

        # Load BLIP Retrieval Model
        text_retrieval = BlipForImageTextRetrieval.from_pretrained('Salesforce/blip-itm-base-flickr')
        self.text_encoder = text_retrieval.text_encoder
        self.visual_encoder = text_retrieval.vision_model
        self.visual_proj = text_retrieval.vision_proj
        self.text_proj = text_retrieval.text_proj
        del text_retrieval
        
        # Freeze weights
        # for param in self.image_model_captioning.parameters():
        #     param.requires_grad = False

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.visual_encoder.parameters():
            param.requires_grad = False

        # for param in self.text_decoder.parameters():
        #     param.requires_grad = False

        # self.image_model_captioning.eval()
        self.text_encoder.eval()
        self.visual_encoder.eval()
        # self.text_decoder.eval()


    # def forward(self,
    #             device: torch.DeviceObjType,
    #             pixel_values: torch.FloatTensor,
    #             input_ids: Optional[torch.LongTensor] = None,
    #             attention_mask: Optional[torch.LongTensor] = None,
    #             labels: Optional[torch.LongTensor] = None):
    #
    #     bs = pixel_values.size(0)
    #
    #     # Params
    #     return_dict = True
    #     output_attentions = True
    #     output_hidden_states = True
    #
    #     # Put models and tensors on correct device
    #     input_ids = input_ids.to(device)
    #     attention_mask = attention_mask.to(device)
    #     pixel_values = pixel_values.to(device)
    #     labels = labels.to(device)
    #     self.image_model_captioning = self.image_model_captioning.to(device)
    #     self.image_model_retrieval = self.image_model_retrieval.to(device)
    #     self.text_model = self.text_model.to(device)
    #     self.text_decoder = self.text_decoder.to(device)
    #
    #     # Get image caption output weights
    #     image_outputs_captioning = self.image_model_captioning(
    #         pixel_values=pixel_values,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #     # Get image retrieval output weights
    #     image_outputs_retrieval = self.image_model_retrieval(
    #         pixel_values=pixel_values,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #     image_embeds = image_outputs_retrieval.last_hidden_state
    #     image_proj = F.normalize(self.image_proj(image_embeds[:, 0, :]), dim=-1)
    #     # image_retrieval_avg = image_embeds[:,1:,:].mean(dim=1, keepdim=True)
    #     # image_retrieval_avg = image_retrieval_avg.view(bs, -1)
    #     # image_retrieval
    #
    #
    #     # Get text encoder output weights
    #     text_encoder_outputs = self.text_model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #     text_embeds = text_encoder_outputs.last_hidden_state
    #     text_proj = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
    #
    #
    #     image_embeds_retrieval = image_outputs_retrieval[0]
    #     image_atts_retrieval = torch.ones(image_embeds_retrieval.size()[:-1],dtype=torch.long)
    #
    #     # Get fused encoder output weights
    #     fused_outputs = self.text_model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         encoder_hidden_states=image_embeds_retrieval,
    #         encoder_attention_mask=image_atts_retrieval,
    #         return_dict=return_dict,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #     )
    #
    #     image_embeds_captioning = image_outputs_captioning[0]
    #     image_atts_captioning = torch.ones(image_embeds_retrieval.size()[:-1], dtype=torch.long)
    #
    #     fused_proj = F.normalize(self.text_proj(fused_outputs.last_hidden_state[:, 0, :]), dim=-1)
    #
    #     # Get text decoder output weights
    #     outputs = self.text_decoder(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         encoder_hidden_states=image_embeds_captioning,
    #         encoder_attention_mask=image_atts_captioning,
    #         labels=labels,
    #         return_dict=return_dict,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         reduction="mean",
    #     )
    #
    #     # Return all weights
    #     return image_outputs_captioning, image_outputs_retrieval, image_proj, text_proj, text_encoder_outputs, fused_outputs, outputs

if __name__ == '__main__':
    test = BlipParent()
