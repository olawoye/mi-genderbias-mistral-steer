"""Single source of truth for model ids, layers and paths (Mistral repo)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = Path("/content/drive/MyDrive")


def on_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists()


def _drive_or_local(env_var: str, sub: str, local: Path) -> Path:
    """$env_var if set, else a Drive path on Colab (survives runtime resets), else repo-local."""
    env = os.environ.get(env_var)
    if env:
        return Path(env)
    return DRIVE_ROOT / sub if on_colab() else local


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
    # Checkpoints + results go to Drive on Colab. `checkpoint.export_results` copies the
    # final JSON/PNGs into the repo's results/ folder for committing.
    results_dir: Path = field(default_factory=lambda: _drive_or_local("MIBIAS_RESULTS", "mibias_results/mistral", REPO_ROOT / "results"))
    cache_dir: Path = field(default_factory=lambda: _drive_or_local("MIBIAS_CACHE", "mibias_cache/mistral", REPO_ROOT / ".cache"))


def get_config(smoke: bool | None = None) -> Config:
    """smoke=True (or env MIBIAS_SMOKE=1) swaps in the tiny model for local tests."""
    cfg = Config()
    if smoke is None:
        smoke = os.environ.get("MIBIAS_SMOKE") == "1"
    if smoke:
        cfg = replace(cfg, model_id=cfg.smoke_model_id, n_layers=None, d_model=None,
                      cache_dir=REPO_ROOT / ".cache" / "smoke",
                      results_dir=REPO_ROOT / ".cache" / "smoke_results")
    return cfg
