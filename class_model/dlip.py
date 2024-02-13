import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torch import nn
from torchvision.transforms import Normalize, ToPILImage

from class_model_help.tie import tie_encoder_decoder_weights
from class_model.bert import BertConfig
from class_model_help.gpu import concat_all_gather
from class_model_help.pretrain import init_blip_pretrain, init_vit, init_bert_decoder, init_bert_encoder, init_tokenizer, init_blip_caption

# DLIP Base Model
class DLIP(nn.Module):
    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------INITIALIZE FUNCTION---------------------------------------------------------------------------
    # PyTorch Initialization Function (i.e., initialize models, create model layers, set config, etc.)
    def __init__(self,
                 image_size=None,
                 bert_pretrain=None,
                 bert=None,
                 vit=None,
                 blip=None,
                 embed_dim=256,
                 queue_size=1600,
                 momentum=0.995,
                 prompt='a picture of ',
                 print=None,
                 multi=None):

        '''
        Training DLIP model for retrieval and captioning

        image_size (int): Image size (image_size x image_size - must match vit model)
        bert_pretrain (bool_str): Pretrain BERT or train BERT from scratch
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
        self.tokenizer = init_tokenizer()

        # Initialize BERT Config
        self.bert = bert
        if bert_pretrain == "True":
            self.bert_pretrain = True
        else:
            self.bert_pretrain = False

        encoder_config = None
        decoder_config = None
        bert_config = None

        if self.bert_pretrain == False:
            if self.bert == "tiny":
                bert_config = {
                                "architectures": ["BertModel"],
                                "hidden_size": 192,
                                "hidden_act": "gelu",
                                "initializer_range": 0.02,
                                "vocab_size": 30524,
                                "hidden_dropout_prob": 0.1,
                                "num_attention_heads": 12,
                                "model_type": "bert",
                                "type_vocab_size": 2,
                                "max_position_embeddings": 512,
                                "layer_norm_eps": 1e-12,
                                "num_hidden_layers": 4,
                                "intermediate_size": 512,
                                "pad_token_id": 0,
                                "attention_probs_dropout_prob": 0.1,
                                "encoder_width": 768,
                                "add_cross_attention": "True"
                              }

            encoder_config = BertConfig(**bert_config)
            encoder_config.encoder_width = image_emb_size

            decoder_config = BertConfig(**bert_config)
            decoder_config.num_attention_heads = 12
            decoder_config.encoder_width = image_emb_size
            self.bert_config = encoder_config

        # Initialize BERT Text Encoder Model (for all losses)
        self.text_encoder, text_emb_size = init_bert_encoder(bert, encoder_config, pretrain=self.bert_pretrain)

        # Initialize BERT Text Decoder Model (for LM loss)
        self.text_decoder, text_emb_size = init_bert_decoder(bert, decoder_config, pretrain=self.bert_pretrain)

        # Tie encoder and decoder
        tie_encoder_decoder_weights(self.text_encoder, self.text_decoder.bert, "", "/attention")

        # Initialize Image Embedding Projection Linear Layer and Text Embedding Projection Linear Layer (for all losses)
        self.image_proj = nn.Linear(image_emb_size, embed_dim)
        self.text_proj = nn.Linear(text_emb_size, embed_dim)

        # Initialize Image Embedding to Text Embedding Projection Linear Layer (for ITM and LM loss - only required for pretrained bert)
        if self.bert_pretrain:
            self.image_text_proj = nn.Linear(image_emb_size, text_emb_size)

        # Initialize Loss ITM Linear Layer (for ITM loss)
        self.itm_head = nn.Linear(text_emb_size, 2)

        # Initialize Momentum Encoder Models for Image and Text (for ITA loss)
        self.image_encoder_m, _ = init_vit(vit)
        self.image_proj_m = nn.Linear(image_emb_size, embed_dim)
        self.text_encoder_m, _ = init_bert_encoder(bert, encoder_config, pretrain=self.bert_pretrain)
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

        self.criterion_kl = nn.KLDivLoss(reduction='sum')

        # Set Text Prompt
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # Print caption
        self.print = print
        if self.print == "True":
            self.process, self.caption, _ = init_blip_caption('base')
            self.inverse_normalize = Normalize(
                mean=[-m / s for m, s in
                      zip((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))],
                std=[1 / s for s in (0.26862954, 0.26130258, 0.27577711)]
            )
            self.to_pil = ToPILImage()

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------HELPER FUNCTION--------------------------------------------------------------------------------
    # Display Batch Captions and Images (to verify data)
    def display_batch(self, caption, image):
        n_images = len(image)
        cols = 5
        rows = (n_images + cols - 1) // cols
        fig = plt.figure(figsize=(15, 3 * rows))

        for i, (img, caption) in enumerate(zip(image, caption)):
            ax = fig.add_subplot(rows, cols, i + 1)
            # Apply the inverse normalization and convert to PIL for display
            img = self.inverse_normalize(img)
            img = self.to_pil(img)
            ax.imshow(img)
            ax.set_title(caption, fontsize=9)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

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
        assert self.queue_size % batch_size == 0, f"Assertion failed: queue_size ({self.queue_size}) must be divisible by batch_size ({batch_size})."

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------LOSS FUNCTION---------------------------------------------------------------------------------
    # Image-Text Contrastive Loss (ITA Loss)
    def ita_loss(self, alpha, idx, image, image_feat, text, text_feat):
        # Get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.image_encoder_m(image)
            image_feat_m = F.normalize(self.image_proj_m(image_embeds_m.last_hidden_state[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            if self.bert_pretrain:
                text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                    return_dict=True)
            else:
                text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                    return_dict=True, mode='text')

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

    # Image-Text Matching Loss (ITM Loss)
    def itm_loss(self, sim_i2t, sim_t2i, image, image_attn, image_embeds, text, text_attn):
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

        # with torch.no_grad():
        #     probs = torch.softmax(vl_output, dim=1)
        #     _, predictions = torch.max(probs, dim=1)
        #     correct_predictions = torch.eq(predictions, itm_labels.to(image.device))
        #     accuracy = correct_predictions.sum().float() / itm_labels.size(0)
        #     print(accuracy)

        return fused_attention, fused_hidden_states, loss_itm, output_pos, encoder_input_ids

    # Captioning Loss (LM Loss)
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
        return decoder_output, decoder_targets, loss_lm, decoder_input_ids

    # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------GENERATE FUNCTION------------------------------------------------------------------------------
    # Pytorch Class For Model Generation
    def generate(self, image, sample=True, num_beams=5, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        # Get image embeddings
        image_embeds = self.image_encoder(image)
        image_embeds = image_embeds.last_hidden_state.to(image.device)

        if self.bert_pretrain == False:
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
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------SETUP IMAGE AND VERIFY DATA-----------------------------------------------------------------------
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        # # Convert image to be the same for dlip and blip (must be formatted for blip)
        # img_min = image.min()
        # img_max = image.max()
        # image = (image - img_min) / (img_max - img_min)
        # image.mul_(255).clamp_(0, 255)
        # image = image.to(image.device, dtype=torch.float32)

        # Print Caption
        if self.print == "True":
            # Get image embedding and feature
            self.display_batch(caption, image)

            # # Select the first image in the batch (or another index)
            # single_image = image[0].cpu()
            #
            # # Convert to PIL Image
            # transform = T.ToPILImage()
            # single_image_pil = transform(single_image)
            #
            # # Display the image
            # plt.imshow(single_image_pil)
            # plt.axis('off')
            # plt.show()
            #
            # print_input = self.process(images=image, return_tensors="pt").to(image.device)
            # with torch.no_grad():
            #     print_captions = self.caption.generate(**print_input)
            # decoded_captions = [self.process.decode(caption, skip_special_tokens=True) for caption in print_captions]
            #
            # print("-" * 60)
            # print("Actual Captions:")
            # print(caption)
            # print("-" * 60)
            # print("BLIP Generated Captions:")
            # print(decoded_captions)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------IMAGE-TEXT EMBEDDING RETRIEVAL-----------------------------------------------------------------
        image_dict = self.image_encoder(image, output_hidden_states=True, output_attentions=True)
        image_embeds = image_dict.last_hidden_state.to(device=image.device)
        image_attn = torch.ones(image_embeds.size()[:-1]).to(device=image.device)

        image_feat = F.normalize(self.image_proj(image_embeds[:, 0, :]), dim=-1)

        # Get text embedding and feature
        text = (self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(image.device))
        text_attn = text.attention_mask.to(device=image.device)

        if self.bert_pretrain:
            text_output = self.text_encoder(text.input_ids.to(image.device), attention_mask=text_attn, output_hidden_states=True, output_attentions=True, return_dict=True)
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1).to( device=image.device)
        else:
            text_output = self.text_encoder(text.input_ids.to(image.device), attention_mask=text_attn, output_hidden_states=True, output_attentions=True, return_dict=True, mode='text')
            text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1).to(device=image.device)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------IMAGE-TEXT CONTRASTIVE LOSS (ITA)-------------------------------------------------------------------
        # ITA Loss
        idx, loss_ita, sim_i2t, sim_t2i = self.ita_loss(alpha, idx, image, image_feat, text, text_feat)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -----------------------------------------------------------------------------IMAGE-TEXT MATCH LOSS (ITM)-----------------------------------------------------------------------
        # ITM Loss
        fused_attention, fused_hidden_states, loss_itm, output_pos, encoder_input_ids = self.itm_loss(sim_i2t, sim_t2i, image, image_attn, image_embeds, text, text_attn)

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------CAPTION LOSS (LM)-----------------------------------------------------------------------------
        # LM Loss
        decoder_output, decoder_targets, loss_lm, decoder_input_ids = self.lm_loss(image, image_attn, image_embeds, text, text_attn)

        # Store HR and ATTN
        last_states_decoder = decoder_output.hidden_states[-1][:, 0, :]
        prediction_scores = decoder_output.logits

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------START DISTILLATION LOSS--------------------------------------------------------------------------
        loss_dist = torch.tensor(0.0).to(image.device)

        if self.blip != "None":
            # Run and get the teacher ITA, ITM Loss and LM Outputs
            self.blip.visual_encoder.to(image.device)
            self.blip.text_encoder.to(image.device)
            self.blip.text_decoder.to(image.device)
            self.blip.vision_proj.to(image.device)
            self.blip.text_proj.to(image.device)

            image_embeds_teacher = self.blip.visual_encoder(image)
            image_atts_teacher = torch.ones(image_embeds_teacher.size()[:-1], dtype=torch.long).to(image.device)
            image_feat_teacher = F.normalize(self.blip.vision_proj(image_embeds_teacher[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_teacher.t(), self.image_queue.clone().detach()], dim=1)

            text_output_teacher = self.blip.text_encoder(input_ids=text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
            text_feat_teacher = F.normalize(self.blip.text_proj(text_output_teacher.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_teacher.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_t = image_feat_teacher @ text_feat_all / self.temp
            sim_t2i_t = text_feat_teacher @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_t.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_t, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_t, dim=1) + (1 - alpha) * sim_targets


            encoder_output_teacher = self.blip.text_encoder(input_ids=encoder_input_ids,
                                                          attention_mask=text.attention_mask,
                                                          encoder_hidden_states=image_embeds_teacher,
                                                          encoder_attention_mask=image_atts_teacher,
                                                          output_attentions=True,
                                                          output_hidden_states=True,
                                                          return_dict=True)

            decoder_output_teacher = self.blip.text_decoder(input_ids=decoder_input_ids,
                                                            attention_mask=text.attention_mask,
                                                            encoder_hidden_states=image_embeds_teacher,
                                                            encoder_attention_mask=image_atts_teacher,
                                                            output_attentions=True,
                                                            output_hidden_states=True,
                                                            labels=decoder_targets.to(image.device),
                                                            return_dict=True)

            prediction_scores_teacher = decoder_output_teacher.logits
            last_states_decoder_teacher = decoder_output_teacher.hidden_states[-1][:, 0, :]


            ###============== Dist CLIP  ===================###
            ##=======  cosine similar loss  =======##
            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp
            loss_i2t_d = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
            loss_t2i_d = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
            loss_ita_dist = (loss_i2t_d + loss_t2i_d) / 2

            criterion_cos = nn.CosineEmbeddingLoss(0.25)
            bs = image.size(0)
            t = torch.ones(bs).to(image.device)
            loss_img_dist = criterion_cos(image_feat, image_feat_teacher, t)
            loss_text_dist = criterion_cos(text_feat, text_feat_teacher, t)

            vl_feat_s = self.text_proj(output_pos.last_hidden_state[:, 0, :])
            vl_feat_t = self.blip.text_proj(encoder_output_teacher.last_hidden_state[:, 0, :])
            loss_vl_dist = criterion_cos(vl_feat_s, vl_feat_t, t)

            vl_decoder_s = self.text_proj(last_states_decoder)
            vl_decoder_t = self.blip.text_proj(last_states_decoder_teacher)
            loss_vl_dec_dist = criterion_cos(vl_decoder_s, vl_decoder_t, t)

            ##======= logits KL loss  =======##a
            temp = 1.0
            vocab_size = int(self.bert_config.vocab_size)
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            logits = shifted_prediction_scores.view(-1, vocab_size)

            shifted_prediction_scores_tea = prediction_scores_teacher[:, :-1, :].contiguous()
            logits_tea = shifted_prediction_scores_tea.view(-1, vocab_size)
            bs_logits = logits.size(0)
            loss_kl = (1.0 / bs_logits) * self.criterion_kl(F.log_softmax(logits / temp, dim=1),
                                                            F.softmax(logits_tea / temp, dim=1))

            ##======= Self Attentions loss  =======##
            temp_score = 0.05
            attention = output_pos.attentions[-1]
            attention_tea = encoder_output_teacher.attentions[-1]
            attention_score = attention.view(-1, 30)
            attention_score_tea = attention_tea.view(-1, 30)

            bs_atten = attention_score.size(0)
            loss_Sattn = (1.0 / bs_atten) * self.criterion_kl(F.log_softmax(attention_score / temp_score, dim=1),
                                                              F.softmax(attention_score_tea / temp_score, dim=1))

            ##======= Cross Attentions loss  =======##
            temp_score = 0.05

            x_attention = output_pos.cross_attentions[-1]
            x_attention_tea = encoder_output_teacher.cross_attentions[-1]
            x_attention_score = x_attention.view(-1, 197)
            x_attention_score_tea = x_attention_tea.view(-1, 197)
            bs_atten = x_attention_score.size(0)
            loss_Xattn = (1.0 / bs_atten) * self.criterion_kl(F.log_softmax(x_attention_score / temp_score, dim=1),
                                                              F.softmax(x_attention_score_tea / temp_score, dim=1))

            ##======= Decoder Self Attentions loss  =======##
            temp_score = 0.05
            dx_attention = decoder_output.attentions[-1]
            dx_attention_tea = decoder_output_teacher.attentions[-1]
            dx_attention_score = dx_attention.view(-1, 30)
            dx_attention_score_tea = dx_attention_tea.view(-1, 30)
            bs_atten = dx_attention_score.size(0)
            loss_DXattn = (1.0 / bs_atten) * self.criterion_kl(F.log_softmax(dx_attention_score / temp_score, dim=1),
                                                               F.softmax(dx_attention_score_tea / temp_score, dim=1))

            loss_dist += loss_img_dist
            loss_dist += loss_text_dist
            loss_dist += loss_vl_dist
            loss_dist += loss_vl_dec_dist
            loss_dist += loss_ita_dist
            loss_dist += loss_kl
            loss_dist += loss_Sattn
            loss_dist += loss_Xattn
            loss_dist += loss_DXattn

        # Return Losses
        return loss_ita, loss_itm, loss_lm, loss_dist
