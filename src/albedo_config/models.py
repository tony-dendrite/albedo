from __future__ import annotations

JUDGE_MODELS: tuple[str, ...] = ("z-ai/glm-5.2",)

JUDGE_PROVIDER_PINS: dict[str, dict[str, object]] = {
    model: {"allow_fallbacks": True, "quantizations": ["fp8"], "order": ["baidu", "streamlake"]}
    for model in JUDGE_MODELS
}

EVALUATOR_MODEL = "z-ai/glm-5.2"
EVALUATOR_PROVIDERS = "streamlake,baidu"
SOTA_MODELS = "z-ai/glm-5.2"
SIMULATION_MODEL = "deepseek/deepseek-v4-flash-0731"
SIMULATION_PROVIDERS = "baidu,streamlake,deepinfra"
ENGY_MODELS = "z-ai/glm-5.2,deepseek/deepseek-v4-flash-0731"
