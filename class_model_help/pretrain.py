import torch

from transformers import ViTModel, BertTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForImageTextRetrieval

from class_model.blip_parent import BlipParent
from class_model.bert import BertModel, BertLMHeadModel
from class_model.blip_pretrain_parent import blip_pretrain

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------INITIALIZE PRETRAINED MODELS--------------------------------------------------------------------------
# Initialize BLIP Pretrained (used for distillation loss)
def init_blip_pretrain(size, model='capfilt'):
    assert size in ['base'], "blip parameter must be base"
    # Parent Processor
    blip_caption_emb_size = 768
    blip_caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    if model == 'retrieval':
        blip_parent = BlipParent()
    else:
        blip_parent = blip_pretrain('https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth')
    return blip_caption_processor, blip_parent, blip_caption_emb_size

# Initialize BLIP Caption
def init_blip_caption(size):
    assert size in ['base', 'large'], "blip parameter must be base or large"
    # Parent Processor
    processor_caption = None
    # Parent Model
    blip_caption = None
    # Parent Embedding Output Size (embeddings are the last hidden state of the model)
    caption_emb_size = None

    if size == 'large':
        processor_caption = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float32)
        caption_emb_size = 1024
    elif size == 'base':
        processor_caption = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_caption = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float32)
        caption_emb_size = 768

    return processor_caption, blip_caption, caption_emb_size

# Initialize BLIP Retrieval
def init_blip_retrieval(size):
    assert size in ['base', 'large'], "blip parameter must be base or large"
    # Parent Processor
    processor_retrieval = None
    # Parent Model
    blip_retrieval = None
    # Parent Embedding Output Size (embeddings are the last hidden state of the model)
    retrieval_emb_size = None

    if size == 'large':
        processor_retrieval = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
        blip_retrieval = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr")
        retrieval_emb_size = 1024
    elif size == 'base':
        processor_retrieval = BlipProcessor.from_pretrained("Salesforce/blip-itm-large-flickr")
        blip_retrieval = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-flickr")
        retrieval_emb_size = 1024

    return processor_retrieval, blip_retrieval, retrieval_emb_size

# Initialize BERT Encoder
def init_bert_encoder(bert, encoder_config, pretrain=False, encoder_width=768):
    assert bert in ['tiny', 'mini', 'small', 'medium', 'large', 'base'], "bert parameter must be tiny, mini, small, medium, large, or base"
    # Text Encoder
    text_encoder = None
    # Text Embedding Output Size (embeddings are the last hidden state of the model)
    text_emb_size = None

    if bert == 'base':
        text_emb_size = 768
        text_encoder = BertModel.from_pretrained('bert-base-uncased', encoder_width=encoder_width, add_cross_attention=True, ignore_mismatched_sizes=True, add_pooling_layer=False)
    elif bert == 'large':
        text_emb_size = 1024
        text_encoder = BertModel.from_pretrained('bert-large-uncased', encoder_width=encoder_width, add_cross_attention=True, ignore_mismatched_sizes=True, add_pooling_layer=False)
    elif bert == 'medium':
        text_emb_size = 512
        if pretrain:
            text_encoder = BertModel.from_pretrained('prajjwal1/bert-medium', encoder_width=encoder_width, add_cross_attention=True, ignore_mismatched_sizes=True, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
    elif bert == 'small':
        text_emb_size = 512
        if pretrain:
            text_encoder = BertModel.from_pretrained('prajjwal1/bert-small', encoder_width=encoder_width, add_cross_attention=True, ignore_mismatched_sizes=True, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
    elif bert == 'mini':
        text_emb_size = 256
        if pretrain:
            text_encoder = BertModel.from_pretrained('prajjwal1/bert-mini', encoder_width=encoder_width, add_cross_attention=True, ignore_mismatched_sizes=True, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)
    elif bert == 'tiny':
        text_emb_size = 192
        if pretrain:
            text_encoder = BertModel.from_pretrained('prajjwal1/bert-tiny', encoder_width=encoder_width, add_cross_attention=True, ignore_mismatched_sizes=True, add_pooling_layer=False)
        else:
            text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)

    # Resize embeddings to match tokenizer
    text_encoder.resize_token_embeddings(30524)
    if pretrain:
        return text_encoder, text_emb_size
    else:
        return text_encoder, encoder_config.hidden_size

# Initialize BERT Decoder
def init_bert_decoder(bert, decoder_config, pretrain=False):
    assert bert in ['tiny', 'mini', 'small', 'medium', 'large', 'base'], "bert parameter must be tiny, mini, small, medium, large, or base"
    # Text Decoder
    text_decoder = None
    # Text Embedding Output Size (embeddings are the last hidden state of the model)
    text_emb_size = None

    if bert == 'base':
        text_emb_size = 768
        if pretrain:
            text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased')
        else:
            text_decoder = BertLMHeadModel(config=decoder_config)
    elif bert == 'large':
        text_emb_size = 1024
        if pretrain:
            text_decoder = BertLMHeadModel.from_pretrained('bert-large-uncased')
        else:
            text_decoder = BertLMHeadModel(config=decoder_config)

    elif bert == 'medium':
        text_emb_size = 512
        if pretrain:
            text_decoder = BertLMHeadModel.from_pretrained('prajjwal1/bert-medium')
        else:
            text_decoder = BertLMHeadModel(config=decoder_config)
    elif bert == 'small':
        text_emb_size = 512
        if pretrain:
            text_decoder = BertLMHeadModel.from_pretrained('prajjwal1/bert-small')
        else:
            text_decoder = BertLMHeadModel(config=decoder_config)
    elif bert == 'mini':
        text_emb_size = 256
        if pretrain:
            text_decoder = BertLMHeadModel.from_pretrained('prajjwal1/bert-mini')
        else:
            text_decoder = BertLMHeadModel(config=decoder_config)
    elif bert == 'tiny':
        text_emb_size = 128
        if pretrain:
            text_decoder = BertLMHeadModel.from_pretrained('prajjwal1/bert-tiny')
        else:
            text_decoder = BertLMHeadModel(config=decoder_config)

    # Resize embeddings to match tokenizer
    text_decoder.resize_token_embeddings(30524)
    if pretrain:
        return text_decoder, text_emb_size
    else:
        return text_decoder, decoder_config.hidden_size
# Initialize Vit Model
def init_vit(vit):
    assert vit in ['tiny', 'small', 'base', 'large'], "vit parameter must be tiny, small, base, or large"
    # Image encoder
    image_encoder = None
    # Image Embedding Output Size (embeddings are the last hidden state of the model)
    image_emb_size = None

    if vit == 'large':
        image_emb_size = 1024
        image_encoder = ViTModel.from_pretrained('google/vit-large-patch16-224')
    if vit == 'base':
        image_emb_size = 768
        image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
    if vit == 'small':
        image_emb_size = 384
        image_encoder = ViTModel.from_pretrained('facebook/deit-small-patch16-224')
        # image_encoder = ViTModel.from_pretrained('WinKawaks/vit-small-patch16-224')
    elif vit == 'tiny':
        image_emb_size = 192
        image_encoder = ViTModel.from_pretrained('facebook/deit-tiny-patch16-224')

    return image_encoder, image_emb_size

# Initialize BERT Tokenizer
def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer