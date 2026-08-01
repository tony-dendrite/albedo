from __future__ import annotations

import asyncio
import json

import httpx

from albedo_eval_service.judge_config import JudgeSettings
from albedo_eval_service.judge_openrouter import OpenRouterJudgeClient


def test_openrouter_payload_respects_provider_structured_output_support():
    payloads = asyncio.run(_capture_payloads())

    plain_payload = payloads[0]
    assert plain_payload["model"] == "z-ai/glm-5.2"
    assert "order" not in plain_payload["provider"]
    assert plain_payload["provider"]["quantizations"] == ["fp8"]
    assert plain_payload["provider"]["allow_fallbacks"] is True
    assert plain_payload["provider"]["require_parameters"] is True
    assert "response_format" not in plain_payload

    schema_payload = payloads[1]
    assert schema_payload["model"] == "qwen/qwen3.5-397b-a17b"
    assert "order" not in schema_payload["provider"]
    assert schema_payload["provider"]["quantizations"] == ["fp8"]
    assert schema_payload["provider"]["allow_fallbacks"] is True
    assert schema_payload["provider"]["require_parameters"] is True
    assert schema_payload["response_format"]["type"] == "json_schema"


async def _capture_payloads():
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content.decode()))
        raw = json.dumps({"answers": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": raw}}]})

    settings = JudgeSettings(openrouter_api_key="test-key")
    client = OpenRouterJudgeClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.score(model="z-ai/glm-5.2", messages=[{"role": "user", "content": "x"}])
        await client.score(
            model="qwen/qwen3.5-397b-a17b",
            messages=[{"role": "user", "content": "x"}],
            response_schema={"type": "object", "properties": {"answers": {"type": "array"}}},
        )
    finally:
        await client.aclose()
    return payloads


def test_provider_order_rotates_across_retries():
    orders = asyncio.run(_capture_orders_under_failures())
    assert orders[0] == ["A", "B", "C"]
    assert orders[1] == ["B", "C", "A"]
    assert orders[2] == ["C", "A", "B"]


async def _capture_orders_under_failures():
    orders = []

    def handler(request: httpx.Request) -> httpx.Response:
        orders.append(json.loads(request.content.decode())["provider"]["order"])
        if len(orders) < 3:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    settings = JudgeSettings(openrouter_api_key="test-key", retry_count=2, retry_backoff_seconds=0)
    client = OpenRouterJudgeClient(settings)
    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url=settings.openrouter_base_url.rstrip("/"),
        transport=httpx.MockTransport(handler),
    )
    try:
        await client.complete(
            model="z-ai/glm-5.2",
            messages=[{"role": "user", "content": "x"}],
            provider={"order": ["A", "B", "C"], "allow_fallbacks": True, "quantizations": ["fp8"]},
        )
    finally:
        await client.aclose()
    return orders
