"""
FastConformer encoder for Canary-MLX using parakeet-mlx implementation.

This module wraps the parakeet-mlx Conformer implementation to match
the Canary-Qwen encoder architecture.

Based on parakeet-mlx (https://github.com/senstella/parakeet-mlx)
Licensed under Apache 2.0
"""

from conformer_parakeet import Conformer, ConformerArgs


def create_conformer_from_config(config_dict: dict) -> Conformer:
    """
    Create FastConformer encoder from Canary configuration.

    Args:
        config_dict: Configuration from model config.json

    Returns:
        Conformer instance from parakeet-mlx
    """
    # Extract encoder config (from perception.encoder in Canary config)
    perception_cfg = config_dict.get("perception", {})
    encoder_cfg = perception_cfg.get("encoder", {})

    # Build ConformerArgs matching parakeet-mlx structure
    args = ConformerArgs(
        feat_in=encoder_cfg.get("feat_in", 80),  # Mel bins
        n_layers=encoder_cfg.get("n_layers", 17),
        d_model=encoder_cfg.get("d_model", 1024),
        n_heads=encoder_cfg.get("n_heads", 8),
        ff_expansion_factor=encoder_cfg.get("ff_expansion_factor", 4),
        subsampling_factor=encoder_cfg.get("subsampling_factor", 8),
        self_attention_model=encoder_cfg.get("self_attention_model", "rel_pos"),
        subsampling=encoder_cfg.get("subsampling", "dw_striding"),
        conv_kernel_size=encoder_cfg.get("conv_kernel_size", 9),
        subsampling_conv_channels=encoder_cfg.get("subsampling_conv_channels", 256),
        pos_emb_max_len=encoder_cfg.get("pos_emb_max_len", 5000),
        causal_downsampling=encoder_cfg.get("causal_downsampling", False),
        use_bias=encoder_cfg.get("use_bias", True),
        xscaling=encoder_cfg.get("xscaling", True),
        subsampling_conv_chunking_factor=encoder_cfg.get("subsampling_conv_chunking_factor", 1),
        att_context_size=encoder_cfg.get("att_context_size", None),
    )

    return Conformer(args)


# Expose the Conformer class for direct use if needed
FastConformerEncoder = Conformer
