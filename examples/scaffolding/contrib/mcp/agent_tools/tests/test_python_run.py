# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from pathlib import Path
from unittest import mock

import pytest
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from tools import python_run as python_impl


@pytest.mark.asyncio
async def test_python_run_missing_key():
    """When E2B_API_KEY is not set, return a clear error message."""
    with mock.patch.dict(os.environ, {"E2B_API_KEY": ""}, clear=False):
        # Force re-read of env in run(); the module may have already read it
        result = await python_impl.run("print(1)")
    assert "E2B_API_KEY" in result
    assert "not set" in result or "Error" in result


@pytest.mark.asyncio
async def test_python_run_with_key():
    """With E2B_API_KEY set, run simple code and return stdout/results. Skips if no key."""
    key = (os.environ.get("E2B_API_KEY") or "").strip()
    if not key:
        pytest.skip("E2B_API_KEY not set; set in .env to run real sandbox test")

    code = "print('hello'); 2 + 3"
    result = await python_impl.run(code, timeout_sec=30)
    assert not result.strip().startswith("Error:") and "Error running code:" not in result
    # E2B returns stdout and/or results
    assert "hello" in result or "5" in result or "stdout" in result or "results" in result
    print("\n" + "=" * 60)
    print("Python sandbox output:")
    print(result)
    print("=" * 60 + "\n")
