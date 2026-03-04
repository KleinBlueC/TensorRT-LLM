# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Run Python code in a sandboxed environment (E2B) for computation and visualization."""

import asyncio
import os


def _run_in_sandbox(code: str, timeout_sec: float = 60.0) -> str:
    """Execute code in E2B sandbox (sync). Returns formatted stdout, stderr, and results."""
    from e2b_code_interpreter import Sandbox

    sbx = Sandbox.create()
    try:
        execution = sbx.run_code(code, timeout=timeout_sec)
        stdout_list = execution.logs.stdout if execution.logs else []
        stderr_list = execution.logs.stderr if execution.logs else []
        stdout = "\n".join(stdout_list).strip() if isinstance(stdout_list, list) else (stdout_list or "").strip()
        stderr = "\n".join(stderr_list).strip() if isinstance(stderr_list, list) else (stderr_list or "").strip()
        results = execution.results or []

        parts = []
        if getattr(execution, "error", None):
            err = execution.error
            err_msg = f"{getattr(err, 'name', 'Error')}: {getattr(err, 'value', err)}"
            if getattr(err, "traceback", None):
                err_msg += "\n" + err.traceback
            parts.append("error:\n" + err_msg)
        if stdout:
            parts.append("stdout:\n" + stdout)
        if stderr:
            parts.append("stderr:\n" + stderr)
        if results:
            result_strs = []
            for r in results:
                if hasattr(r, "text") and r.text:
                    result_strs.append(r.text)
                else:
                    result_strs.append(str(r))
            parts.append("results:\n" + "\n".join(result_strs))
        if not parts:
            parts.append("(no output)")
        return "\n\n".join(parts)
    finally:
        try:
            sbx.kill()
        except Exception:
            pass


async def run(code: str, timeout_sec: float = 60.0) -> str:
    """Run Python code in a sandboxed environment for computational tasks.

    Uses E2B Code Interpreter: standard libraries and data/visualization (e.g. numpy,
    pandas, matplotlib) are available. All outputs are explicitly captured and
    returned (stdout, stderr, and cell results) for clear result communication.

    Args:
        code: Python code to execute (Jupyter-style; single statements or blocks).
        timeout_sec: Maximum execution time in seconds (default 60).

    Returns:
        Combined stdout, stderr, and results as a single string. If E2B_API_KEY
        is not set, returns an error message instead.
    """
    if not (os.environ.get("E2B_API_KEY") or "").strip():
        return (
            "Error: E2B_API_KEY is not set. Set it in .env to use the Python sandbox "
            "(e.g. from https://e2b.dev)."
        )

    try:
        return await asyncio.to_thread(_run_in_sandbox, code, timeout_sec)
    except Exception as e:
        return f"Error running code: {e!s}"
