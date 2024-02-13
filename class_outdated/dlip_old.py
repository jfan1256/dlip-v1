import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torch import nn

from class_model_help.tie import tie_encoder_decoder_weights
from class_model.bert import BertModel, BertConfig, BertLMHeadModel
from class_model_help.distill import hr_loss, attn_loss
from class_model_help.gpu import concat_all_gather
from class_model_help.pretrain import init_blip_pretrain, init_vit, init_bert_decoder, init_bert_encoder, init_tokenizer, init_blip_caption


# DLIP Base Model
class DLIPOld(nn.Module):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------INITIALIZE FUNCTION---------------------------------------------------------------------------
    # PyTorch Initialization Function (i.e., initialize models, create model layers, set config, etc.)
    def __init__(self,
                 image_size=None,
                 bert=None,
                 vit=None,
                 blip=None,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 prompt='a picture of ',
                 print=None,
                 multi=None):

        '''
        Training DLIP model for retrieval and captioning

        image_size (int): Image size (image_size x image_size - must match vit model)
        bert (str): Size of BERT model (either 'tiny', 'mini', 'small', 'medium', 'large, or 'base')
        vit (str): Size of Vit model (either 'tiny' or 'small')
        blip (str): Size of BLIP model (either 'base' or 'large')
        embed_dim (int): Embeddings dimension to map image embedding and text embedding to the same dimension for comparison
        queue_size (int): Size of dequeue enqueue
        momentum (float): Momentum encoder
        prompt (str): Prompt for training
        print (bool): Print parent caption or not
        multi (bool): Multi-GPU training or not

        Note: Text Embedding and Image Embedding are the Last Hidden State (or hidden representation) of BERT and Vit respectively
        '''

        super().__init__()

        # Assert Statement
        if vit == 'tiny' or vit == 'small' or vit == 'large' or vit == 'base':
            assert image_size == 224, "image_size must be 224 to use tiny, small, base, or, large vit"

        # Blip Parent Model
        blip_emb_size = None

        # Initialize Parent BLIP Models (no need to for dlip_evaluate unless validation evaluation is required)
        self.blip = blip
        if blip != "None":
            self.processor_blip, self.blip, blip_emb_size = init_blip_pretrain(self.blip)

        # Multi-GPU processing or not
        self.multi = multi

        # Initialize Vit Image Encoder Model (for all losses)
        self.vit = vit
        self.image_encoder, image_emb_size = init_vit(vit)

        # Initialize BERT Tokenizer Model (for all losses)
        self.bert = bert
        self.tokenizer = init_tokenizer()

        # Initialize BERT Config
        enc_config = None
        dec_config = None
        text_emb_size = None

        if self.bert == "tiny":
            text_emb_size = image_emb_size
            enc_config = {
                "architectures": ["BertModel"],
                "hidden_size": text_emb_size,
                "hidden_act": "gelu",
                "initializer_range": 0.02,
                "vocab_size": 30524,
                "hidden_dropout_prob": 0.1,
                "num_attention_heads": 16,
                "model_type": "bert",
                "type_vocab_size": 2,
                "max_position_embeddings": 512,
                "layer_norm_eps": 1e-12,
                "num_hidden_layers": 16,
                "intermediate_size": 512,
                "pad_token_id": 0,
                "attention_probs_dropout_prob": 0.1,
                "encoder_width": text_emb_size,
                "add_cross_attention": True
            }

            dec_config = {
                "architectures": ["BertModel"],
                "hidden_size": text_emb_size,
                "hidden_act": "gelu",
                "initializer_range": 0.02,
                "vocab_size": 30524,
                "hidden_dropout_prob": 0.1,
                "num_attention_heads": 16,
                "model_type": "bert",
                "type_vocab_size": 2,
                "max_position_embeddings": 512,
                "layer_norm_eps": 1e-12,
                "num_hidden_layers": 16,
                "intermediate_size": 512,
                "pad_token_id": 0,
                "attention_probs_dropout_prob": 0.1,
                "encoder_width": text_emb_size,
                "add_cross_attention": True,
                "is_decoder": True
            }
            enc_config = BertConfig(**enc_config)
            enc_config.encoder_width = image_emb_size
            dec_config = BertConfig(**dec_config)
            dec_config.encoder_width = image_emb_size

        # Initialize BERT Text Encoder Model (for all losses)
        if bert == "tiny":
            self.text_encoder = BertModel(config=enc_config, add_pooling_layer=False)
        else:
            self.text_encoder, text_emb_size = init_bert_encoder(bert)

        # Initialize BERT Text Decoder Model (for LM loss)
        if bert == "tiny":
            self.text_decoder = BertLMHeadModel(config=dec_config)
        else:
            self.text_decoder, text_emb_size = init_bert_decoder(bert)

        # Initialize Image Embedding Projection Linear Layer and Text Embedding Projection Linear Layer (for all losses)
        self.image_proj = nn.Linear(image_emb_size, embed_dim)
        self.text_proj = nn.Linear(text_emb_size, embed_dim)

        # Initialize Image Embedding to Text Embedding Projection Linear Layer (for ITM and LM loss)
        if self.bert != "tiny":
            self.image_text_proj = nn.Linear(image_emb_size, text_emb_size)

        # Initialize Loss ITM Linear Layer (for ITM loss)
        self.itm_head = nn.Linear(text_emb_size, 2)

        # Initialize Hidden Representation Projection Linear Layer for Image and Text (for Distillation Loss)
        if blip != "None":
            self.hr_caption_image = nn.Linear(blip_emb_size, image_emb_size)
            self.hr_retrieval_image = nn.Linear(blip_emb_size, image_emb_size)
            self.hr_caption_dec = nn.Linear(blip_emb_size, text_emb_size)
            self.hr_retrieval_enc = nn.Linear(blip_emb_size, text_emb_size)
            self.hr_fused_enc = nn.Linear(blip_emb_size, text_emb_size)

        # Initialize Momentum Encoder Models for Image and Text (for ITA loss)
        self.image_encoder_m, _ = init_vit(vit)
        self.image_proj_m = nn.Linear(image_emb_size, embed_dim)

        if self.bert == "tiny":
            self.text_encoder_m = BertModel(config=enc_config, add_pooling_layer=False)
        else:
            self.text_encoder_m, _ = init_bert_encoder(bert)
        self.text_proj_m = nn.Linear(text_emb_size, embed_dim)

        self.model_pairs = [[self.image_encoder, self.image_encoder_m],
                            [self.image_proj, self.image_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m]]

        # Copy Params
        self._copy_params()

        # Create queues
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        # Set Text Prompt
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # Tie encoder and decoder
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, "", "/attention")

        # Print caption
        self.print = print
        if self.print == "True":
            self.process, self.caption, _ = init_blip_caption('base')

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------HELPER FUNCTION--------------------------------------------------------------------------------
    # Copy Params for Multi-GPU Computation
    @torch.no_grad()
    def _copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    # Copy Params for Update Momentum
    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    # Dequeue and Enqueue
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat, self.multi)
        text_feats = concat_all_gather(text_feat, self.multi)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------GENERATE FUNCTION------------------------------------------------------------------------------
    # Pytorch Class For Model Generation
    def generate(self, image, sample=True, num_beams=5, max_length=30, min_length=10, top_p=0.9,
                 repetition_penalty=1.0):
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
    def forward(self, image, caption, alpha, idx):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------IMAGE-TEXT EMBEDDING RETRIEVAL-----------------------------------------------------------------
        # Get image embedding and feature
        # self.display_batch(caption, image)

        image_dict = self.image_encoder(image, output_hidden_states=True, output_attentions=True)
        image_embeds = image_dict.last_hidden_state.to(device=image.device)
        image_attn = torch.ones(image_embeds.size()[:-1]).to(device=image.device)
        image_feat = F.normalize(self.image_proj(image_embeds[:, 0, :]), dim=-1)

        # Get text embedding and feature
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
            image.device)
        text_attn = text.attention_mask.to(device=image.device)

        if self.bert == "tiny":
            text_output = self.text_encoder(text.input_ids.to(image.device), attention_mask=text_attn,
                                            output_hidden_states=True, output_attentions=True, return_dict=True,
                                            mode='text')
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1).to(
                device=image.device)
        else:
            text_output = self.text_encoder(text.input_ids.to(image.device), attention_mask=text_attn,
                                            output_hidden_states=True, output_attentions=True, return_dict=True)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1).to(
                device=image.device)
            # Map image_embeds's embed_size -> [batch_size, num_states, embed_size] -> to be the same size as text_embeds's embed size
            # Example: image_embeds => [3, 197, 192] --> [3, 197, 128]
            image_embeds = self.image_text_proj(image_embeds).to(image.device)
            image_attn = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # Store HR and ATTN
        hr_image_encoder = image_dict.hidden_states
        attn_image_encoder = image_dict.attentions[-1]
        hr_text_encoder = text_output.hidden_states
        attn_text_encoder = text_output.attentions[-1]

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------IMAGE-TEXT CONTRASTIVE LOSS (ITA)-----------------------------------------------------------------
        # Set up idx
        idx, loss_ita, sim_i2t, sim_t2i = self.ita_loss(alpha, idx, image, image_feat, text, text_attn, text_feat)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------IMAGE-TEXT MATCH LOSS (ITM)-----------------------------------------------------------------------
        # Encode text input id with self.tokenizer
        fused_attention, fused_hidden_states, loss_itm = self.itm_loss(sim_i2t, sim_t2i, idx, image, image_attn,
                                                                       image_embeds, image_feat,
                                                                       text, text_attn, text_feat)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------CAPTION LOSS (LM)---------------------------------------------------------------------------------
        # Encode text input id with self.tokenizer
        decoder_output, decoder_targets, loss_lm = self.lm_loss(image, image_attn, image_embeds, text, text_attn)

        # Store HR and ATTN
        hr_text_decoder = decoder_output.hidden_states
        attn_text_decoder = decoder_output.attentions[-1]

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------START DISTILLATION LOSS-----------------------------------------------------------------------------
        loss_attn = torch.tensor(0)
        loss_hr = torch.tensor(0)
        if self.blip != "None":
            # Create compatible image for parent model
            img_min = image.min()
            img_max = image.max()
            parent_image = (image - img_min) / (img_max - img_min)
            parent_image.mul_(255).clamp_(0, 255)
            parent_image = parent_image.to(image.device, dtype=torch.uint8)

            # PretrainBLIP: Get hidden states (make sure to stack) and attention layer
            inputs_caption = self.processor_blip(images=parent_image, text=caption, return_tensors="pt",
                                                 padding='max_length', truncation=True, max_length=35).to(image.device)
            output = self.blip.forward(device=image.device, labels=decoder_targets, **inputs_caption)

            # Print Caption
            if self.print == "True":
                # Select the first image in the batch (or another index)
                single_image = parent_image[0].cpu()

                # Convert to PIL Image
                transform = T.ToPILImage()
                single_image_pil = transform(single_image)

                # Display the image
                plt.imshow(single_image_pil)
                plt.axis('off')
                plt.show()

                print_input = self.process(images=parent_image, return_tensors="pt").to(image.device)
                with torch.no_grad():
                    print_captions = self.caption.generate(**print_input)
                decoded_captions = [self.process.decode(caption, skip_special_tokens=True) for caption in
                                    print_captions]

                print("-" * 60)
                print("Actual Captions:")
                print(caption)
                print("-" * 60)
                print("BLIP Generated Captions:")
                print(decoded_captions)

            attn_teacher_caption_image = output.caption_image_attentions[-1]
            attn_teacher_retrieval_image = output.retrieval_image_attentions[-1]
            attn_teacher_decoder = output.decoder_attentions[-1]
            attn_teacher_encoder = output.text_enc_attentions[-1]
            attn_teacher_fused = output.fused_attentions[-1]

            # Stack Dlip HR
            hr_teacher_caption_image = torch.stack(output.caption_image_hidden_states)
            hr_teacher_retrieval_image = torch.stack(output.retrieval_image_hidden_states)
            hr_teacher_decoder = torch.stack(output.decoder_hidden_states)
            hr_teacher_encoder = torch.stack(output.text_enc_hidden_states)
            hr_teacher_fused_encoder = torch.stack(output.fused_hidden_states)
            hr_image_encoder = torch.stack(hr_image_encoder)
            hr_text_encoder = torch.stack(hr_text_encoder)
            hr_text_decoder = torch.stack(hr_text_decoder)
            hr_fused = torch.stack(fused_hidden_states)

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ---------------------------------------------------------------------------HIDDEN REPRESENTATION LOSS (HR)---------------------------------------------------------------------
            # Hidden Representation Loss
            loss_hr_caption_image = hr_loss(self.hr_caption_image, hr_teacher_caption_image, hr_image_encoder,
                                            image.device)
            loss_hr_retrieval_image = hr_loss(self.hr_retrieval_image, hr_teacher_retrieval_image, hr_image_encoder,
                                              image.device)
            loss_hr_decoder = hr_loss(self.hr_caption_dec, hr_teacher_decoder, hr_text_decoder, image.device)
            loss_hr_encoder = hr_loss(self.hr_retrieval_enc, hr_teacher_encoder, hr_text_encoder, image.device)
            loss_hr_fused = hr_loss(self.hr_fused_enc, hr_teacher_fused_encoder, hr_fused, image.device)
            loss_hr = 0.5 * loss_hr_caption_image + 0.5 * loss_hr_retrieval_image + loss_hr_decoder + loss_hr_encoder + loss_hr_fused
            # loss_hr = loss_hr_retrieval_image + loss_hr_decoder + loss_hr_encoder + loss_hr_fused

            # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------ATTENTION LOSS (ATTN)------------------------------------------------------------------------
            # Attention Loss
            loss_attn_caption_image = attn_loss(attn_teacher_caption_image, attn_image_encoder)
            loss_attn_retrieval_image = attn_loss(attn_teacher_retrieval_image, attn_image_encoder)
            loss_attn_encode = attn_loss(attn_teacher_encoder, attn_text_encoder)
            loss_attn_decode = attn_loss(attn_teacher_decoder, attn_text_decoder)
            loss_attn_fused = attn_loss(attn_teacher_fused, fused_attention)
            loss_attn = 0.5 * loss_attn_caption_image + 0.5 * loss_attn_retrieval_image + loss_attn_encode + loss_attn_decode + loss_attn_fused
            # loss_attn = loss_attn_retrieval_image + loss_attn_encode + loss_attn_decode + loss_attn_fused

        # Return Losses
        return loss_ita, loss_itm, loss_lm, loss_hr, loss_attn

    def display_batch(self, caption: torch.Tensor, image: torch.Tensor):
        cols = 3  # Number of columns in the grid
        rows = (len(image) + cols - 1) // cols  # Calculate rows needed, round up
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.subplots_adjust(hspace=0.5)
        # Flatten the array of axes for easy iteration
        axs = axs.flatten()
        for i in range(len(axs)):
            if i < len(image):
                axs[i].imshow(image[i].detach().cpu().permute(1, 2, 0))
                axs[i].set_title(caption[i], fontsize=10)
                axs[i].axis('off')  # Hide the axes ticks
            else:
                axs[i].axis('off')  # Hide unused subplots

    def ita_loss(self, alpha, idx, image, image_feat, text, text_attn, text_feat):
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.image_encoder_m(image)
            image_feat_m = F.normalize(self.image_proj_m(image_embeds_m.last_hidden_state[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            if self.bert == 'tiny':
                text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                    return_dict=True, mode='text')
            else:
                text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                    return_dict=True)
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)
        return idx, loss_ita, sim_i2t, sim_t2i

    def lm_loss(self, image, image_attn, image_embeds, text, text_attn):
        decoder_input_ids = text.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.tokenizer.pad_token_id, -100)
        decoder_output = self.text_decoder(decoder_input_ids.to(image.device),
                                           attention_mask=text_attn,
                                           encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_attn,
                                           output_attentions=True,
                                           output_hidden_states=True,
                                           labels=decoder_targets.to(image.device),
                                           return_dict=True,
                                           )
        loss_lm = decoder_output.loss
        return decoder_output, decoder_targets, loss_lm

    def itm_loss(self, sim_i2t, sim_t2i, idx, image, image_attn, image_embeds, image_feat, text, text_attn, text_feat):
        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        # Forward the positive image-text pair
        bs = image.size(0)
        output_pos = self.text_encoder(encoder_input_ids.to(image.device),
                                       attention_mask=text_attn,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_attn,
                                       output_attentions=True,
                                       output_hidden_states=True,
                                       return_dict=True,
                                       )

        fused_attention = output_pos.attentions[-1]
        fused_hidden_states = output_pos.hidden_states

        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1) + 1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1) + 1e-4
            weights_i2t.fill_diagonal_(0)
        # Select a negative image (from same rank) for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        # Select a negative text (from same rank) for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_attn, image_attn], dim=0)

        output_neg = self.text_encoder(text_ids_all.to(image.device),
                                       attention_mask=text_atts_all.to(image.device),
                                       encoder_hidden_states=image_embeds_all.to(image.device),
                                       encoder_attention_mask=image_atts_all.to(image.device),
                                       return_dict=True,
                                       )
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0)
        loss_itm = F.cross_entropy(vl_output.to(image.device), itm_labels.to(image.device))
        return fused_attention, fused_hidden_states, loss_itm