"""Utility script to create a Mamba2Config given the size of the model."""

from typing import Dict

from mamba2.configuration_mamba2 import Mamba2Config
from mamba2hybrid import Mamba2HybridConfig
from dataclasses import dataclass, astuple
import torch

@dataclass
class ConfParams:
    d_model: int
    n_layers: int
    head_dim: int = 128


MAMBA2_CONFIGS_KWARGS: Dict[str, ConfParams] = {
    'Mamba130M': ConfParams(d_model=768, n_layers=24),
    'Mamba370M': ConfParams(d_model=1024, n_layers=48),
    'Mamba780M': ConfParams(d_model=1536, n_layers=48),
    'Mamba1B': ConfParams(d_model=2048, n_layers=48),
    'Mamba3B': ConfParams(d_model=2560, n_layers=64),
    'Mamba7B': ConfParams(d_model=4096, n_layers=64),
}


MAMBA2HYBRID_CONFIGS_KWARGS: Dict[str, ConfParams] = {
    'Mamba2Hybrid800M': ConfParams(d_model=1024, n_layers=48, head_dim=64),
    'Mamba2Hybrid8B': ConfParams(d_model=4096, n_layers=56, head_dim=128),
}


CONFIGS_KWARGS: Dict[str, ConfParams] = {
    **MAMBA2_CONFIGS_KWARGS,
    **MAMBA2HYBRID_CONFIGS_KWARGS,
}


def get_config(name: str, vocab_size, rmsnorm_within_groups=True, n_groups=8, sequence_parallel_enabled=False,
               dtype=torch.float32):
    d_model, n_layers, head_dim = astuple(CONFIGS_KWARGS[name])
    if name in MAMBA2_CONFIGS_KWARGS.keys():
        config = Mamba2Config(
            vocab_size=vocab_size,
            hidden_size=d_model,
            head_dim=head_dim,
            num_heads=(d_model * 2) // head_dim,
            num_hidden_layers=n_layers,
            tie_word_embeddings=True,
            use_cache=False,
            n_groups=n_groups,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
            rmsnorm_within_groups=rmsnorm_within_groups,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

    elif name in MAMBA2HYBRID_CONFIGS_KWARGS.keys():
        if 'nvidia' in name:
            pass
        else: # For the tekken tokenizer
            config_dict = {
                'vocab_size': vocab_size,
                "bos_token_id": 0,
                "eos_token_id": 0,
                "pad_token_id": 0,
                "tie_word_embeddings": True,
                "hidden_size": d_model,
                "head_dim": head_dim,
                "num_hidden_layers": n_layers,
                "n_groups": n_groups,
                "rmsnorm_within_groups": rmsnorm_within_groups,
                "num_heads": (d_model * 2) // head_dim,
                "sequence_parallel_enabled": sequence_parallel_enabled,
                "dtype": dtype,
                "use_cache": False,
                
            } # TODO: check configs are correct

        config = Mamba2HybridConfig(**config_dict)
    else:
        raise ValueError(f'Unrecognized config name: {name}')

    return config

