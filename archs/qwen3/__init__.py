"""Architecture shim for any Qwen3ForCausalLM checkpoint.

Importing this ensures `transformers` is loaded so HF Auto* resolves any
Qwen3 model (1.7B, 4B, 7B, …) without trust_remote_code=True.
"""
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

__all__ = ["AutoConfig", "AutoModelForCausalLM", "AutoTokenizer"]
