# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

import pytest
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from tools import visit as visit_impl


@pytest.mark.asyncio
async def test_visit_empty_url():
    """Empty or missing URL returns an error message."""
    result = await visit_impl.run("")
    assert "Error" in result or "empty" in result.lower()


@pytest.mark.asyncio
async def test_visit_real_url():
    """Real Jina Reader fetch: get content from a known URL and print. No API key required."""
    url = "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc6"
    # url = "https://example.com"
    max_chars = 5000
    result = await visit_impl.run(url, max_chars=max_chars)
    assert not result.startswith("Error:"), f"Visit returned error: {result}"
    assert "Visit failed" not in result, f"Visit failed: {result}"
    assert len(result) > 0, "Expected non-empty content"
    print("\n" + "=" * 60)
    print(f"URL: {url}")
    print(f"max_chars: {max_chars}")
    print("=" * 60)
    print("Content (excerpt):")
    print(result[:2000] + ("..." if len(result) > 2000 else ""))
    print("=" * 60 + "\n")


@pytest.mark.asyncio
async def test_visit_truncation():
    """Content beyond max_chars is truncated with a marker.`"""
    url = "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc6"
    result = await visit_impl.run(url, max_chars=100)
    assert "truncated" in result or len(result) <= 150
    assert len(result) > 0
