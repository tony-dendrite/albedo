"""Single settings tree - the only place environment variables are read."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChainSettings(BaseSettings):
    # Chain ingest: subtensor endpoint, reveal filter, and legacy hotkey backfill cutoff.
    model_config = SettingsConfigDict(env_prefix="CHAIN_", extra="ignore", env_ignore_empty=True)

    netuid: int = 97
    network: str = "finney"
    poll_interval_s: float = 12.0
    start_block: int = 0
    ignore_commits_to_block: int = Field(default=0, validation_alias="IGNORE_COMMITS_TO_BLOCK")
    mock: bool = Field(default=False, validation_alias="CHAIN_MOCK")


class DbSettings(BaseSettings):
    # One asyncpg pool for every backend loop; falls back to ALBEDO_POSTGRES_* parts.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_POSTGRES_", extra="ignore", env_ignore_empty=True
    )

    user: str = "albedo"
    password: str = "albedo"
    db: str = "albedo"
    host: str = "127.0.0.1"
    host_port: int = 65432

    @property
    def dsn(self) -> str:
        # ALBEDO_EVAL_DATABASE_URL wins when set (same contract as the old tree).
        import os

        explicit = os.environ.get("ALBEDO_EVAL_DATABASE_URL", "").strip()
        if explicit:
            return explicit
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.host_port}/{self.db}"


class S3Settings(BaseSettings):
    # Hippius S3 - fault reports, guard detections, fingerprint corpus, sanity results.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_S3_", extra="ignore", env_ignore_empty=True
    )

    bucket: str = ""
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.bucket and self.endpoint and self.access_key and self.secret_key)


class OpenSearchSettings(BaseSettings):
    # Fingerprint dedup corpus (kNN prefilter + exact rerank).
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_OPENSEARCH_", extra="ignore", env_ignore_empty=True
    )

    url: str = ""
    user: str = ""
    password: str = ""
    index: str = "albedo_fingerprints"


class ValidationSettings(BaseSettings):
    # Hippius validation worker: manifest -> dtype -> download -> index -> arch -> dedup.
    model_config = SettingsConfigDict(env_prefix="ALBEDO_", extra="ignore", env_ignore_empty=True)

    model_cache_dir: str = "/root/miners_models"
    sim_threshold: float = 0.95
    max_knn_dim: int = 16000
    knn_candidates: int = 20
    hv_max_attempts: int = 5
    hv_lease_seconds: int = 600
    arch_spec: str = ""  # path override; default = packaged architecture_spec.json
    mock: bool = Field(default=False, validation_alias="HIPPIUS_VALIDATION_MOCK")


class EvalSettings(BaseSettings):
    # Eval dispatcher: dataset pin, lease, retries, artifact addressing.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_EVAL_", extra="ignore", env_ignore_empty=True
    )

    worker_id: str = "eval-dispatcher"
    remote_auth_token: str = ""
    dataset_version: str = "AlienKevin/SWE-ZERO-12M-trajectories"
    dataset_manifest_uri: str = ""
    dataset_manifest_hash: str = ""
    dataset_manifest_path: str = ""
    sample_count: int = 128
    max_turns_per_sample: int = 10
    sampling_algo: str = "swe-zero-multi-source-sample-v1"
    judge_config_hash: str = ""
    artifact_bucket: str = "albedo-artifacts"
    artifact_prefix: str = "s3://albedo-artifacts"
    lease_seconds: int = 1800
    max_retry_count: int = 3
    remote_base_url: str = "http://127.0.0.1:18090"
    remote_event_poll_seconds: float = 5.0
    remote_event_timeout_seconds: float = 30.0
    dispatch_poll_seconds: float = 5.0
    judge_count: int = 3


class SanityDispatchSettings(BaseSettings):
    # Sanity dispatcher: claim -> sample -> GPU worker -> judge gate.
    model_config = SettingsConfigDict(
        env_prefix="SANITY_DISPATCH_", extra="ignore", env_ignore_empty=True
    )

    worker_id: str = "sanity-dispatcher"
    remote_auth_token: str = ""
    remote_base_url: str = "http://127.0.0.1:19100"
    consensus: bool = False
    dataset_manifest_path: str = ""
    dataset_manifest_hash: str = ""
    dataset_root: str = ""
    sample_count: int = 3
    max_turns_per_sample: int = 10
    gen_max_tokens: int = 32768
    lease_seconds: int = 600
    max_retry_count: int = 5
    min_free_gpus: int = 1
    judge_models: str = Field(default="", validation_alias="SANITY_JUDGE_MODELS")
    injection_recheck_temperature: float = Field(
        default=0.2, validation_alias="SANITY_INJECTION_RECHECK_TEMPERATURE"
    )


class SanityRemoteSettings(BaseSettings):
    # Sanity GPU worker: stateless vLLM generation + heuristics; no DB, no judge keys.
    model_config = SettingsConfigDict(
        env_prefix="SANITY_REMOTE_", extra="ignore", env_ignore_empty=True
    )

    auth_token: str = ""
    host_id: str = "sanity-remote-1"
    api_port: int = 9100
    vllm_port: int = 9101
    gpu_ids: str = "0"
    gpu_util: float = 0.5
    vllm_dtype: str = "bfloat16"
    # Sized for the 35B reference model (JIT compile + full OCI pull), not a 4B toy.
    vllm_startup_s: int = 1200
    download_timeout_s: int = 1800
    model_cache_dir: str = "/root/miners_models"
    # 0 = use the canonical genesis max_position_embeddings (262144), matching prod f72163d.
    max_model_len: int = 0
    gen_max_tokens: int = 32768
    mock_auto_result: bool = False


class JudgeSettings(BaseSettings):
    # Judge ensemble: binary yes/no-question judging via OpenRouter (evaluator + 3 judges).
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_JUDGE_", extra="ignore", env_ignore_empty=True
    )

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api"
    request_timeout_seconds: float = 90.0
    retry_count: int = 5
    retry_backoff_seconds: float = 1.5
    parse_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 768
    max_concurrency_per_model: int = 128
    min_valid_fraction: float = 0.8
    evaluator_model: str = "z-ai/glm-5.2"
    evaluator_providers: str = ""
    num_questions: int = 50
    question_max_tokens: int = 16000
    answer_max_tokens: int = 8000
    question_prep_ttl_seconds: float = 1800.0


class RemoteEvalSettings(BaseSettings):
    # Eval GPU worker: 8-GPU duel, canonical config pinning, artifact spool, score bridge.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_REMOTE_", extra="ignore", env_ignore_empty=True
    )

    auth_token: str = ""
    host_id: str = "remote-eval-1"
    api_port: int = Field(
        default=8090,
        validation_alias=AliasChoices("ALBEDO_REMOTE_EVAL_API_PORT", "ALBEDO_REMOTE_API_PORT"),
    )
    gpu_count: int = 8
    accelerator_type: str = ""
    mock_auto_verdict: bool = False
    mock_challenger_won: bool = False
    # Scored mock duel: fabricated generations, REAL score-bridge/judge scoring + verdict math.
    mock_scored_duel: bool = False
    dataset_root: str = ""
    previous_king_gpu_ids: str = "0,1,2,3"
    challenger_gpu_ids: str = "4,5,6,7"
    max_new_tokens: int = 1024
    enforce_eager: bool = False
    use_canonical_model_config: bool = True
    resolve_model_artifacts: bool = True
    model_cache_dir: str = "/tmp/albedo-remote-models"
    model_download_concurrency: int = 8
    artifact_spool_dir: str = "/tmp/albedo-remote-artifacts"
    scoring_min_valid_fraction: float = 0.8
    scoring_timeout_seconds: float = 300.0
    upload_artifacts: bool = True
    cleanup_local_artifacts: bool = False
    s3_endpoint_url: str = ""
    s3_region: str = ""
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_session_token: str = ""


class ScoreBridgeSettings(BaseSettings):
    # Backend-side WS client that dials the eval GPU and answers scoring requests in-process.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_SCORE_BRIDGE_", extra="ignore", env_ignore_empty=True
    )

    remote_ws_url: str = "ws://127.0.0.1:18090/score-bridge"
    remote_auth_token: str = ""
    request_timeout_seconds: float = 300.0
    reconnect_min_seconds: float = 1.0
    reconnect_max_seconds: float = 30.0
    ping_interval_seconds: float = 20.0
    ping_timeout_seconds: float = 20.0
    websocket_max_size_bytes: int = 2_147_483_648


class WeightSettings(BaseSettings):
    # Weight setter: wallet, rate limit, dereg burn, mock mode.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_WEIGHT_", extra="ignore", env_ignore_empty=True
    )

    coldkey: str = ""
    hotkey: str = ""
    wallet_path: str = ""
    network: str = "finney"
    netuid: int = 97
    set_rate_blocks: int = 100
    poll_seconds: float = 12.0
    burn_uid: int = 0
    mock: bool = False
    worker_id: str = "weight-setter"


class MonitorSettings(BaseSettings):
    # Dashboard publisher.
    model_config = SettingsConfigDict(env_prefix="ALBEDO_", extra="ignore", env_ignore_empty=True)

    monitor_interval_s: float = 2.0
    mock: bool = Field(default=False, validation_alias="ALBEDO_MONITOR_MOCK")
    dashboard_netuid: int = 97
    dashboard_artifact_base_url: str = "https://s3.hippius.com"
    dashboard_history_limit: int = 200
    dashboard_model_filter: str = "qwen3.6-35b"
    dashboard_data_dir: str = "website/data"


class ReignSettings(BaseSettings):
    # Set-reign worker: coronation claim identity and pacing.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_SET_REIGN_", extra="ignore", env_ignore_empty=True
    )

    worker_id: str = "set-reign-worker"
    poll_seconds: float = 5.0
    lease_seconds: int | None = None  # None = fall back to eval.lease_seconds


class SlackSettings(BaseSettings):
    # Best-effort Slack error alerts shared by every backend loop.
    model_config = SettingsConfigDict(
        env_prefix="ALBEDO_SLACK_ERROR_", extra="ignore", env_ignore_empty=True
    )

    webhook_url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "ALBEDO_SLACK_ERROR_WEBHOOK_URL",
            "ALBEDO_JUDGE_SLACK_ERROR_WEBHOOK_URL",
            "ALBEDO_REMOTE_SLACK_ERROR_WEBHOOK_URL",
        ),
    )
    username: str = "Albedo Eval Alerts"
    icon_url: str = "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
    timeout_seconds: float = 10.0
    dedupe_seconds: float = 300.0
    env_label: str = Field(default="", validation_alias="ALBEDO_SLACK_ERROR_ENV")


class Settings(BaseSettings):
    # Root aggregate; construct once via get_settings().
    model_config = SettingsConfigDict(extra="ignore", env_ignore_empty=True)

    hippius_hub_token: str = ""

    chain: ChainSettings = ChainSettings()
    db: DbSettings = DbSettings()
    s3: S3Settings = S3Settings()
    opensearch: OpenSearchSettings = OpenSearchSettings()
    validation: ValidationSettings = ValidationSettings()
    eval: EvalSettings = EvalSettings()
    sanity: SanityDispatchSettings = SanityDispatchSettings()
    sanity_remote: SanityRemoteSettings = SanityRemoteSettings()
    judge: JudgeSettings = JudgeSettings()
    remote_eval: RemoteEvalSettings = RemoteEvalSettings()
    score_bridge: ScoreBridgeSettings = ScoreBridgeSettings()
    weights: WeightSettings = WeightSettings()
    monitor: MonitorSettings = MonitorSettings()
    reign: ReignSettings = ReignSettings()
    slack: SlackSettings = SlackSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Cached singleton so every module sees one consistent env snapshot.
    import os

    token = os.environ.get("HIPPIUS_HUB_TOKEN", "")
    return Settings(hippius_hub_token=token)
