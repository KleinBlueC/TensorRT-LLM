# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""SWE-Bench tool implementations: docker exec, file edit, finish, switch_container, run_tests, get_patch, cleanup."""

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .state import (
    CONTAINER_AGENT_LOGS,
    CONTAINER_LOGS,
    CONTAINER_TESTS,
    CONTAINER_TESTBED,
    SWEBenchState,
    get_dataset_path,
)

logger = logging.getLogger(__name__)


def _docker_exec(state: SWEBenchState, command: str, timeout: float = 120.0) -> Dict[str, Any]:
    if not state.container_name or not state.container_running:
        return {
            "output": "Error: No container running. Call __swebench_switch_container first.",
            "exit_code": -1,
        }
    try:
        result = subprocess.run(
            ["docker", "exec", state.container_name, "bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout or ""
        if result.stderr:
            output += "\n" + result.stderr
        if len(output) > 50000:
            output = output[:50000] + "\n<response clipped>"
        return {"output": output.strip() if output else "(no output)", "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"output": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        logger.error("Docker exec error: %s", e)
        return {"output": f"Error: {str(e)}", "exit_code": 1}


def _docker_read_file(state: SWEBenchState, path: str) -> str:
    result = _docker_exec(state, f"cat '{path}'", timeout=30)
    if result["exit_code"] != 0:
        raise FileNotFoundError(f"File not found or cannot read: {path}")
    return result["output"]


def _docker_write_file(state: SWEBenchState, path: str, content: str) -> bool:
    import base64
    encoded = base64.b64encode(content.encode()).decode()
    cmd = f"echo '{encoded}' | base64 -d > '{path}'"
    result = _docker_exec(state, cmd, timeout=30)
    return result["exit_code"] == 0


def execute_bash_impl(
    state: SWEBenchState,
    command: str,
    is_input: str = "false",
    timeout: Optional[float] = None,
) -> str:
    actual_timeout = timeout if timeout else 120.0
    full_command = f"cd {state.workspace} && {command}"
    result = _docker_exec(state, full_command, timeout=actual_timeout)
    return json.dumps(result)


def str_replace_editor_impl(
    state: SWEBenchState,
    command: str,
    path: str,
    file_text: Optional[str] = None,
    old_str: Optional[str] = None,
    new_str: Optional[str] = None,
    insert_line: Optional[int] = None,
    view_range: Optional[List[int]] = None,
) -> str:
    try:
        if command == "view":
            check_result = _docker_exec(state, f"test -d '{path}' && echo 'dir' || echo 'file'")
            is_dir = "dir" in (check_result.get("output") or "")
            if is_dir:
                result = _docker_exec(state, f"find '{path}' -maxdepth 2 -not -name '.*' | head -100")
                return result["output"]
            if view_range and len(view_range) >= 2:
                start, end = view_range[0], view_range[1]
                result = _docker_exec(
                    state,
                    f"sed -n '{start},{end}p' '{path}' | cat -n | awk '{{print {start-1}+NR\"\\t\"$0}}'",
                )
            else:
                result = _docker_exec(state, f"cat -n '{path}'")
            if result["exit_code"] != 0:
                return f"Error: File not found: {path}"
            content = result["output"] or ""
            if len(content) > 30000:
                content = content[:30000] + "\n<response clipped>"
            return content

        if command == "create":
            check_result = _docker_exec(state, f"test -f '{path}' && echo 'exists'")
            if "exists" in (check_result.get("output") or ""):
                return f"Error: File already exists: {path}. Use str_replace to edit."
            if file_text is None:
                return "Error: file_text is required for create command"
            dir_path = str(Path(path).parent)
            _docker_exec(state, f"mkdir -p '{dir_path}'")
            if _docker_write_file(state, path, file_text):
                return f"File created successfully: {path}"
            return f"Error: Failed to create file: {path}"

        if command == "str_replace":
            if old_str is None:
                return "Error: old_str is required for str_replace command"
            try:
                content = _docker_read_file(state, path)
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            if path not in state.edit_history:
                state.edit_history[path] = []
            state.edit_history[path].append(content)
            count = content.count(old_str)
            if count == 0:
                return f"Error: old_str not found in {path}. Make sure it matches exactly."
            if count > 1:
                return f"Error: old_str found {count} times in {path}. Add more context to make it unique."
            new_content = content.replace(old_str, new_str or "", 1)
            if _docker_write_file(state, path, new_content):
                return f"Successfully replaced string in {path}"
            return f"Error: Failed to write file: {path}"

        if command == "insert":
            if insert_line is None:
                return "Error: insert_line is required for insert command"
            if new_str is None:
                return "Error: new_str is required for insert command"
            try:
                content = _docker_read_file(state, path)
            except FileNotFoundError:
                return f"Error: File not found: {path}"
            lines = content.split("\n")
            if path not in state.edit_history:
                state.edit_history[path] = []
            state.edit_history[path].append(content)
            insert_idx = min(insert_line, len(lines))
            new_lines = new_str.split("\n")
            for i, line in enumerate(new_lines):
                lines.insert(insert_idx + i, line)
            new_content = "\n".join(lines)
            if _docker_write_file(state, path, new_content):
                return f"Successfully inserted {len(new_lines)} line(s) after line {insert_line} in {path}"
            return f"Error: Failed to write file: {path}"

        if command == "undo_edit":
            if path not in state.edit_history or not state.edit_history[path]:
                return f"Error: No edit history for {path}"
            previous_content = state.edit_history[path].pop()
            if _docker_write_file(state, path, previous_content):
                return f"Successfully reverted last edit to {path}"
            return f"Error: Failed to revert file: {path}"

        return f"Error: Unknown command '{command}'. Allowed: view, create, str_replace, insert, undo_edit"
    except Exception as e:
        logger.error("Error in str_replace_editor: %s", e)
        return json.dumps({"error": str(e)})


def finish_impl(state: SWEBenchState, message: str) -> str:
    state.task_finished = True
    state.task_result = message
    patch = ""
    try:
        diff_result = _docker_exec(
            state, f"cd {state.workspace} && git diff --no-color HEAD", timeout=60
        )
        patch = diff_result["output"] or ""
        if not patch.strip() or diff_result["exit_code"] != 0:
            cached = _docker_exec(
                state, f"cd {state.workspace} && git diff --no-color --cached", timeout=60
            )
            if (cached.get("output") or "").strip():
                patch = cached["output"]
        state.generated_patch = patch
        logger.info("[SWEBench] Generated patch: %d characters", len(patch))
    except Exception as e:
        logger.warning("[SWEBench] Failed to generate git diff: %s", e)
    logger.info("[SWEBench] Task finished: %s...", (message or "")[:200])
    return json.dumps({
        "status": "completed",
        "message": message,
        "patch": patch,
        "patch_length": len(patch),
        "timestamp": datetime.now().isoformat(),
    })


def switch_container_impl(
    state: SWEBenchState,
    task_id: str,
    output_dir: Optional[str] = None,
    no_rebuild: bool = True,
) -> str:
    dataset_path = get_dataset_path()
    if state.container_running and state.project_name and state.task_path:
        try:
            logger.info("[SWEBench] Cleaning up old container: %s", state.project_name)
            dc = state.task_path / "docker-compose.yaml"
            if dc.exists():
                subprocess.run(
                    ["docker", "compose", "-f", str(dc), "-p", state.project_name, "down", "-v"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            if state.container_name:
                subprocess.run(
                    ["docker", "rm", "-f", state.container_name],
                    capture_output=True,
                    timeout=30,
                )
            state.container_running = False
        except Exception as e:
            logger.warning("[SWEBench] Failed to cleanup old container: %s", e)

    state.task_finished = False
    state.task_result = None
    state.edit_history = {}
    state.generated_patch = ""

    try:
        task_path = dataset_path / task_id
        if not task_path.exists():
            return json.dumps({
                "success": False,
                "error": f"Task not found: {task_id}",
                "searched_path": str(task_path),
            })
        docker_compose_path = task_path / "docker-compose.yaml"
        if not docker_compose_path.exists():
            return json.dumps({
                "success": False,
                "error": f"docker-compose.yaml not found in {task_path}",
            })
        state.task_id = task_id
        state.task_path = task_path

        logs_base = Path(output_dir) if output_dir else (task_path / "output")
        logs_base.mkdir(parents=True, exist_ok=True)
        state.logs_path = logs_base / "logs"
        state.agent_logs_path = logs_base / "agent-logs"
        state.logs_path.mkdir(parents=True, exist_ok=True)
        state.agent_logs_path.mkdir(parents=True, exist_ok=True)

        suffix = str(uuid.uuid4())[:8]
        safe_id = task_id.replace("__", "_").replace("/", "_")
        state.project_name = f"swebench_{safe_id}_{suffix}"
        state.container_name = None

        env = os.environ.copy()
        env["T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME"] = f"swebench_img_{task_id.replace('__', '_')}"
        env["T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME"] = state.project_name
        env["T_BENCH_TASK_LOGS_PATH"] = str(state.logs_path.absolute())
        env["T_BENCH_CONTAINER_LOGS_PATH"] = CONTAINER_LOGS
        env["T_BENCH_TASK_AGENT_LOGS_PATH"] = str(state.agent_logs_path.absolute())
        env["T_BENCH_CONTAINER_AGENT_LOGS_PATH"] = CONTAINER_AGENT_LOGS
        env["T_BENCH_TEST_DIR"] = CONTAINER_TESTS

        build_flag = [] if no_rebuild else ["--build"]
        cmd = [
            "docker", "compose", "-f", str(docker_compose_path),
            "-p", state.project_name, "up", "-d",
        ] + build_flag
        logger.info("[SWEBench] Starting container: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if result.returncode != 0:
            return json.dumps({
                "success": False,
                "error": f"Failed to start container: {result.stderr}",
                "stdout": result.stdout,
            })

        time.sleep(2)
        ps_result = subprocess.run(
            [
                "docker", "compose", "-f", str(docker_compose_path),
                "-p", state.project_name, "ps", "-q",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        container_id = (ps_result.stdout or "").strip().split("\n")[0]
        if not container_id:
            ps2 = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={state.project_name}"],
                capture_output=True,
                text=True,
            )
            container_id = (ps2.stdout or "").strip()
        if container_id:
            state.container_name = container_id
        else:
            return json.dumps({"success": False, "error": "Failed to get container ID after starting"})
        state.container_running = True
        logger.info("[SWEBench] Container started: %s (project: %s)", state.container_name, state.project_name)
        verify = _docker_exec(state, "pwd && ls -la", timeout=30)
        return json.dumps({
            "success": True,
            "task_id": task_id,
            "container_name": state.container_name,
            "project_name": state.project_name,
            "workspace": state.workspace,
            "verification": (verify.get("output") or "")[:500],
        })
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Timeout starting container"})
    except Exception as e:
        logger.exception("Error initializing task")
        return json.dumps({"success": False, "error": str(e)})


def run_tests_impl(state: SWEBenchState) -> str:
    try:
        if not state.task_path or not state.task_id:
            return json.dumps({"success": False, "error": "No task initialized"})
        tests_config = state.task_path / "tests" / "config.json"
        if tests_config.exists():
            config_content = tests_config.read_text()
            _docker_exec(state, "mkdir -p /tests")
            _docker_write_file(state, "/tests/config.json", config_content)
        run_tests_sh = state.task_path / "run-tests.sh"
        if not run_tests_sh.exists():
            return json.dumps({"success": False, "error": f"run-tests.sh not found in {state.task_path}"})
        test_script = run_tests_sh.read_text()
        result = _docker_exec(
            state,
            f"cd {state.workspace} && bash << 'HEREDOC_END'\n{test_script}\nHEREDOC_END",
            timeout=1800,
        )
        test_output = result["output"] or ""
        exit_code = result["exit_code"]
        resolved = False
        tests_status: Dict[str, Any] = {}
        report: Dict[str, Any] = {}
        try:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.test_spec.test_spec import make_test_spec
            from swebench.harness.utils import load_swebench_dataset
            task_json = state.task_path / "task.json"
            if task_json.exists():
                instance = json.loads(task_json.read_text())
            else:
                dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Verified", "test")
                instance = next((i for i in dataset if i.get("instance_id") == state.task_id), None)
                if not instance:
                    raise ValueError(f"Instance {state.task_id} not found")
            test_spec = make_test_spec(instance)
            diff_result = _docker_exec(
                state, f"cd {state.workspace} && git diff --no-color HEAD", timeout=60
            )
            patch = diff_result["output"] or ""
            if not patch.strip():
                diff_result = _docker_exec(
                    state, f"cd {state.workspace} && git diff --no-color --cached", timeout=60
                )
                patch = diff_result["output"] or ""
            with tempfile.TemporaryDirectory() as temp_dir:
                log_dir = Path(temp_dir) / "logs" / state.task_id.lower()
                log_dir.mkdir(parents=True, exist_ok=True)
                (log_dir / "test_output.txt").write_text(test_output)
                eval_report = get_eval_report(
                    test_spec=test_spec,
                    prediction={"model_patch": patch, "instance_id": state.task_id},
                    include_tests_status=True,
                    test_log_path=str(log_dir / "test_output.txt"),
                )
                if state.task_id in eval_report:
                    ir = eval_report[state.task_id]
                    resolved = ir.get("resolved", False)
                    tests_status = ir.get("tests_status") or {}
                    report = {
                        "resolved": resolved,
                        "patch_exists": ir.get("patch_exists", bool(patch)),
                        "patch_successfully_applied": ir.get("patch_successfully_applied", True),
                        "tests_status": tests_status,
                    }
                    logger.info("[SWE-Bench] Evaluation result: resolved=%s", resolved)
        except ImportError:
            logger.warning("swebench library not available, using simple pass/fail")
            resolved = "PASSED" in test_output and "SWEBench results" in test_output
            report = {"resolved": resolved, "fallback_detection": True}
        except Exception as e:
            logger.warning("Error using swebench grading: %s, using simple detection", e)
            resolved = "PASSED" in test_output and "SWEBench results" in test_output
            report = {"resolved": resolved, "fallback_detection": True, "grading_error": str(e)}
        return json.dumps({
            "success": True,
            "passed": resolved,
            "resolved": resolved,
            "tests_status": tests_status,
            "report": report,
            "output": test_output[-10000:] if len(test_output) > 10000 else test_output,
            "exit_code": exit_code,
        })
    except Exception as e:
        logger.exception("Error running tests")
        return json.dumps({"success": False, "error": str(e)})


def get_patch_impl(state: SWEBenchState) -> str:
    try:
        diff_result = _docker_exec(
            state, f"cd {state.workspace} && git diff --no-color HEAD", timeout=60
        )
        patch = diff_result["output"] or ""
        if not patch.strip():
            diff_result = _docker_exec(
                state, f"cd {state.workspace} && git diff --no-color --cached", timeout=60
            )
            patch = diff_result["output"] or ""
        return json.dumps({"success": True, "patch": patch, "length": len(patch)})
    except Exception as e:
        logger.error("Error getting patch: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def cleanup_impl(state: SWEBenchState) -> str:
    try:
        if state.task_path and state.project_name:
            dc = state.task_path / "docker-compose.yaml"
            subprocess.run(
                ["docker", "compose", "-f", str(dc), "-p", state.project_name, "down", "-v"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if state.container_name:
                subprocess.run(
                    ["docker", "rm", "-f", state.container_name],
                    capture_output=True,
                    timeout=30,
                )
        state.container_running = False
        state.container_name = None
        state.project_name = None
        state.task_id = None
        state.task_path = None
        state.task_finished = False
        state.task_result = None
        state.edit_history = {}
        logger.info("[SWEBench] Task cleaned up")
        return json.dumps({"success": True, "message": "Task cleaned up"})
    except Exception as e:
        logger.error("Error cleaning up: %s", e)
        return json.dumps({"success": False, "error": str(e)})
