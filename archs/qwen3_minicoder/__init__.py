"""Architecture shim for ricdomolm/mini-coder-1.7b.

mini-coder-1.7b is a vanilla Qwen3ForCausalLM checkpoint, so there is no
custom modelling code to vendor. Importing this package registers nothing
new — it just ensures `transformers` is loaded so HF Auto* resolves the king
without `trust_remote_code=True` at load time. The validator's
`chain_config.load_arch()` calls into this module.
"""
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

__all__ = ["AutoConfig", "AutoModelForCausalLM", "AutoTokenizer"]
