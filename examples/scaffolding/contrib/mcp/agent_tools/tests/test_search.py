# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from agent_tools root so TAVILY_API_KEY is available
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# Import the module under test; sys.path is set when running from agent_tools/
from tools import search as search_impl


def _has_tavily_key() -> bool:
    key = os.getenv("TAVILY_API_KEY")
    return bool(key and key.strip())


@pytest.mark.asyncio
async def test_search_missing_api_key():
    """Without TAVILY_API_KEY, run() returns an error message."""
    from unittest.mock import patch

    with patch("os.getenv", return_value=None):
        result = await search_impl.run("test query")
    assert "TAVILY_API_KEY" in result
    assert "not set" in result or "Error" in result


@pytest.mark.asyncio
async def test_search_empty_api_key():
    """With empty TAVILY_API_KEY, run() returns an error message."""
    from unittest.mock import patch

    with patch("os.getenv", return_value=""):
        result = await search_impl.run("test query")
    assert "TAVILY_API_KEY" in result


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_tavily_key(), reason="TAVILY_API_KEY not set in .env")
async def test_search_real_api():
    """Real Tavily API call: search for a query and print the result."""
    query = "TensorRT-LLM NVIDIA"
    result = await search_impl.run(query)
    assert not result.startswith("Error:"), f"API returned error: {result}"
    assert "not set" not in result, "API key was not loaded"
    assert "Tavily search failed" not in result, f"API call failed: {result}"
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print("Answer:")
    print(result)
    print("=" * 60 + "\n")


@pytest.mark.asyncio
@pytest.mark.skipif(not _has_tavily_key(), reason="TAVILY_API_KEY not set in .env")
async def test_search_real_api_second_query():
    """Real Tavily API call with a second query; print the result."""
    query = "Python asyncio tutorial"
    result = await search_impl.run(query)
    if result.startswith("Error:") or "Tavily search failed" in result:
        pytest.fail(f"API returned error: {result}")
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print("Answer:")
    print(result)
    print("=" * 60 + "\n")
    # Either we have content or "No results"
    assert isinstance(result, str) and len(result) > 0
