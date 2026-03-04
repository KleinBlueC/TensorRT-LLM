# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

import pytest
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from tools import scholar as scholar_impl


@pytest.mark.asyncio
async def test_scholar_real_api():
    """Real Google Scholar search (scholarly package): search papers and print the result. No API key required."""
    query = "attention is all you need transformer"
    limit = 3
    result = await scholar_impl.run(query, limit=limit)
    assert not result.startswith("Error:"), f"Search returned error: {result}"
    assert "Google Scholar search failed" not in result, f"Search failed: {result}"
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print(f"Limit: {limit}")
    print("=" * 60)
    print("Result:")
    print(result)
    print("=" * 60 + "\n")


@pytest.mark.asyncio
async def test_scholar_real_api_second_query():
    """Real Google Scholar search with another query; print the result."""
    query = "large language model inference optimization"
    limit = 5
    result = await scholar_impl.run(query, limit=limit)
    if result.startswith("Error:") or "Google Scholar search failed" in result:
        pytest.fail(f"Search returned error: {result}")
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print(f"Limit: {limit}")
    print("=" * 60)
    print("Result:")
    print(result)
    print("=" * 60 + "\n")
    assert isinstance(result, str) and len(result) > 0
