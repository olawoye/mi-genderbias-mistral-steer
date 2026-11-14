"""Single source of truth for model ids, layers and paths (Mistral repo)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Target model (gated on Hugging Face: accept the licence on its model page first)
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"
    n_layers: int | None = 32      # None -> read from model.config at load time
    d_model: int | None = 4096
    dtype: str = "bfloat16"

    # Tiny random-weight Mistral, for CPU smoke tests of the plumbing only (outputs are noise)
    smoke_model_id: str = "hf-internal-testing/tiny-random-MistralForCausalLM"

    # Chosen from notebook 02 (activation patching); None until then.
    steer_layer: int | None = None

    seed: int = 0
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")
    # On Colab, cache heavy activations on Drive so a runtime reset doesn't lose them.
    cache_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("MIBIAS_CACHE", "/content/drive/MyDrive/mibias_cache/mistral")
    ))


def get_config(smoke: bool | None = None) -> Config:
    """smoke=True (or env MIBIAS_SMOKE=1) swaps in the tiny model for local tests."""
    cfg = Config()
    if smoke is None:
        smoke = os.environ.get("MIBIAS_SMOKE") == "1"
    if smoke:
        cfg = replace(cfg, model_id=cfg.smoke_model_id, n_layers=None, d_model=None,
                      cache_dir=REPO_ROOT / ".cache")
    return cfg
