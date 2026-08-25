from __future__ import annotations

import functools

from loguru import logger as log

from albedo_config import get_model_validation_settings

config = get_model_validation_settings()


@functools.lru_cache(maxsize=1)
def get_client():
    from opensearchpy import OpenSearch

    use_ssl = config.OPENSEARCH_URL.lower().startswith("https")
    auth = (config.OPENSEARCH_USER, config.OPENSEARCH_PASSWORD) if config.OPENSEARCH_USER else None
    return OpenSearch(
        hosts=[config.OPENSEARCH_URL],
        http_auth=auth,
        use_ssl=use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
        timeout=30,
    )


def health() -> bool:
    try:
        status = get_client().cluster.health().get("status")
        log.info("opensearch health: {}", status)
        return status in ("green", "yellow")
    except Exception as exc:
        log.warning("opensearch health check failed: {}", exc)
        return False
