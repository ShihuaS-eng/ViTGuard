import ml_collections

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.attention_probs_dropout_prob = 0.0
    config.encoder_stride = 16
    config.hidden_act = 'gelu'
    config.hidden_dropout_prob = 0.0 
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



