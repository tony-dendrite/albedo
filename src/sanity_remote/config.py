
from __future__ import annotations

import sys
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from albedo_eval_service.modelstore.canonical_model_config import canonical_max_model_len


class SanityRemoteSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SANITY_REMOTE_", extra="ignore")

    auth_token: str = ""
    host_id: str = "sanity-remote-local"
    host_role: str = "PRE_EVAL"
    ready: bool = True
    api_port: int = 9100

    gpu_ids: str = "0,1"
    gpu_util: float = 0.95
    vllm_port: int = 9101
    vllm_dtype: str = "bfloat16"
    vllm_startup_s: float = 1200.0
    vllm_python: str = sys.executable
    vllm_quantization: str = ""
    vllm_enforce_eager: bool = True
    vllm_moe_backend: str = ""
    vllm_compile_cache_dir: str = ""
    tensor_parallel_size: int = 2
    data_parallel_size: int = 1
    data_parallel_size_local: int = 0
    cpu_offload_gb: int = 0
    download_timeout_s: float = 3000.0
    model_cache_dir: str = "/root/miners_models"
    max_model_len: int = canonical_max_model_len()
    kv_cache_dtype: str = "auto"
    vllm_limit_mm: str = '{"image": 0, "video": 0}'
    gen_temperature: float = 0.7
    gen_top_p: float = 0.8
    gen_top_k: int = 20
    gen_min_p: float = 0.0
    gen_read_timeout_s: float = 900.0

    mock_auto_result: bool = False
    skip_heuristics: bool = False


@lru_cache
def get_remote_settings() -> SanityRemoteSettings:
    return SanityRemoteSettings()
