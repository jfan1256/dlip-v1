import torch

from torch import nn

from class_model_help.pretrain import init_vit, init_tokenizer, init_bert_decoder

# DLIP Finetune Caption Model
class DLIPCaption(nn.Module):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------INITIALIZE FUNCTION---------------------------------------------------------------------------
    # PyTorch Initialization Function (i.e., initialize models, create model layers, set config, etc.)
    def __init__(self,
                 image_size=None,
                 bert=None,
                 vit=None,
                 prompt='a picture of ',
                 multi=None):

        '''
        Tuning DLIP model to specifically be suited for captioning (this should be run after training DLIP)

        image_size (int): Image size (image_size x image_size - must match vit model)
        bert (str): Size of BERT model (either 'tiny', 'mini', 'small', 'medium', 'large, or 'base')
        vit (str): Size of Vit model (either 'tiny' or 'small')
        blip (str): Size of BLIP model (either 'base' or 'large')
        embed_dim (int): Embeddings dimension to map image embedding and text embedding to the same dimension for comparison
        prompt (str): Prompt for training
        multi (bool): Multi-GPU training or not

        Note: Text Embedding and Image Embedding are the Last Hidden State (or hidden representation) of BERT and Vit respectively
        '''

        super().__init__()

        # Assert Statement
        if vit == 'tiny' or vit == 'small' or vit == 'large' or vit == 'base':
            assert image_size == 224, "image_size must be 224 to use tiny, small, base, or, large vit"

        # Multi-GPU processing or not
        self.multi = multi

        # Initialize Vit Image Encoder Model (for LM loss)
        self.vit = vit
        self.image_encoder, image_emb_size = init_vit(vit)

        # Initialize Bert
        self.bert = bert

        # Initialize BERT Tokenizer Model (for LM loss)
        self.tokenizer = init_tokenizer()

        # Initialize BERT Text Encoder Model (for LM loss)
        self.text_decoder, text_emb_size = init_bert_decoder(bert, )

        # Initialize Image Embedding to Text Embedding Projection Linear Layer (for LM loss)
        self.image_text_proj = nn.Linear(image_emb_size, text_emb_size)

        # Set Text Prompt
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # Create temp
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------GENERATE FUNCTION------------------------------------------------------------------------------
    # Pytorch Class For Model Generation
    def generate(self, image, sample=True, num_beams=5, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        # Get image embeddings
        image_embeds = self.image_encoder(image)
        image_embeds = image_embeds.last_hidden_state.to(image.device)

        if self.bert == "tiny":
            text_image_embeds = image_embeds
            image_attn = torch.ones(text_image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        else:
            # Map image_embeds's embed_size -> [batch_size, num_states, embed_size] -> to be the same size as text_embeds's embed size
            # Example: image_embeds => [3, 197, 192] --> [3, 197, 128]
            text_image_embeds = self.image_text_proj(image_embeds)
            image_attn = torch.ones(text_image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if not sample:
            text_image_embeds = text_image_embeds.repeat_interleave(num_beams, dim=0).to(image.device)

        model_kwargs = {"encoder_hidden_states": text_image_embeds, "encoder_attention_mask": image_attn}

        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids.to(image.device),
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 do_sample=True,
                                                 top_p=top_p,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1,
                                                 **model_kwargs)
        else:
            # beam search
            outputs = self.text_decoder.generate(input_ids=input_ids.to(image.device),
                                                 max_length=max_length,
                                                 min_length=min_length,
                                                 num_beams=num_beams,
                                                 eos_token_id=self.tokenizer.sep_token_id,
                                                 pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=repetition_penalty,
                                                 **model_kwargs)

        # Generate DLIP caption
        dlip_captions = []
        for output in outputs:
            dlip_caption = self.tokenizer.decode(output, skip_special_tokens=True)
            dlip_captions.append(dlip_caption[len(self.prompt):])

        return dlip_captions

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------FORWARD FUNCTION------------------------------------------------------------------------------
    # Pytorch Class For Model Training
    def forward(self, image, caption):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------IMAGE-TEXT EMBEDDING RETRIEVAL-----------------------------------------------------------------
        # Get image embedding and feature
        image_dict = self.image_encoder(image, output_hidden_states=True, output_attentions=True)
        image_embeds = image_dict.last_hidden_state.to(device=image.device)
        image_attn = torch.ones(image_embeds.size()[:-1]).to(device=image.device)

        # Get text embedding and feature
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(image.device)
        text_attn = text.attention_mask.to(device=image.device)

        # Map image_embeds's embed_size -> [batch_size, num_states, embed_size] -> to be the same size as text_embeds's embed size
        # Example: image_embeds => [3, 197, 192] --> [3, 35, 128]
        text_image_embeds = self.image_text_proj(image_embeds)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------CAPTION LOSS (LM)---------------------------------------------------------------------------------
        # Encode text input id with self.tokenizer
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id

        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)

        decoder_output = self.text_decoder(text.input_ids.to(image.device),
                                           attention_mask=text_attn,
                                           encoder_hidden_states=text_image_embeds,
                                           encoder_attention_mask=image_attn,
                                           output_attentions=False,
                                           output_hidden_states=False,
                                           labels=decoder_targets.to(image.device),
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss

        # Return Losses
        return loss_lm