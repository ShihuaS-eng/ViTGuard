#Original code from: https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/configuration_vit_mae.py
import ml_collections


def ViTMAEConfig(ratio = 0.75):
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.intermediate_size = 3072
    config.hidden_act = 'gelu'
    config.hidden_dropout_prob = 0.0 #The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    config.attention_probs_dropout_prob = 0.0
    config.initializer_range = 0.02 
    config.layer_norm_eps = 1e-12
    config.image_size = 224
    config.patch_size = 16
    config.num_channels = 3
    config.qkv_bias = True

    config.decoder_num_attention_heads=16
    config.decoder_hidden_size=512
    config.decoder_num_hidden_layers=8
    config.decoder_intermediate_size=2048
    config.mask_ratio=ratio
    config.norm_pix_loss=False
    
    config.chunk_size_feed_forward=False 
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.attention_probs_dropout_prob = 0.0
    config.encoder_stride = 16
    config.hidden_act = 'gelu'
    config.hidden_dropout_prob = 0.0 #The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
    config.hidden_size = 768
    config.image_size = 224
    config.initializer_range = 0.02
    config.intermediate_size = 3072
    config.layer_norm_eps = 1e-12
    config.num_attention_heads = 12
    config.num_channels = 3
    config.num_hidden_layers = 12
    config.patch_size = 16
    config.qkv_bias = True
    return config

