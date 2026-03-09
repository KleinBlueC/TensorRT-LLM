"""Workspace state and conversion to chat messages (RoleMessage list).

OpenAI Chat Completion supports four roles: system, user, assistant, tool.
- system: global instructions (optional).
- user: human/query (here, the research question).
- assistant: model output (here, report and/or tool_calls).
- tool: result of a tool call (here, tool_response after action).

Workspace.to_messages() returns a List[RoleMessage] for the workspace state.
Conversion to OpenAI API message format is done by the worker.

All attribute access is thread-safe: use the get_*/set_* and append_* methods
only; do not access internal attributes. Each getter/setter holds the lock.
"""
import copy
import json
import os
import threading
from datetime import datetime as dt, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

# Beijing time (UTC+8) for default workspace_id.
BEIJING_TZ = timezone(timedelta(hours=8))

from tensorrt_llm.scaffolding.task import (
    AssistantMessage,
    RoleMessage,
    SystemMessage,
    UserMessage,
)

# Synthetic tool_call_id used when converting Workspace to messages (Workspace
# does not store the original MCP call id).
WORKSPACE_TOOL_CALL_ID = "workspace-tool-call"


class ToolMessage(RoleMessage):
    """Tool role message (OpenAI API tool result). Worker uses to_dict() for API format."""

    def __init__(
        self,
        content: str,
        tool_call_id: str,
        name: str,
        prefix: Optional[str] = None,
    ):
        super().__init__(role="tool", content=content, prefix=prefix)
        self.tool_call_id = tool_call_id
        self.name = name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
        }


class Workspace:
    """Compact state for the iterative research pipeline (Markov representation).

    Consists of three components:
    (1) The original research Question q.
    (2) The evolving Report_{i-1} from the previous round (empty for i = 1).
    (3) The most recent Action_{i-1} and its Tool Response_{i-1} (if i > 1).

    This ensures the Markov property while keeping all information needed
    for decision-making.

    Thread-safe: all reads and writes must go through get_*, set_*, and
    append_* methods. Do not access attributes directly.

    Create one instance per IterResearcher; that instance is shared by
    thinker, reporter, actor, and extractor within the same IterResearcher.
    """

    def __deepcopy__(self, memo: dict) -> "Workspace":
        """Copy state into a new Workspace; preserve same workspace_id so the same
        report file is used (no new file created). Skips init_report_file to avoid
        truncating the file.
        """
        return Workspace(
            question=self.get_question(),
            report=self.get_report(),
            actions=self.get_actions(),
            tool_args=self.get_tool_args(),
            tool_responses=self.get_tool_responses(),
            iteration=self.get_iteration(),
            workspace_id=self._workspace_id,
            save_reports=self._save_reports,
            _skip_init_report_file=True,
        )

    def __getstate__(self) -> dict:
        """Omit _lock so pickle/unpickle does not fail; restore in __setstate__."""
        return {k: v for k, v in self.__dict__.items() if k != "_lock"}

    def __setstate__(self, state: dict) -> None:
        """Restore state and create a new RLock after unpickle."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def __init__(
        self,
        question: Optional[str] = None,
        report: Optional[str] = None,
        actions: Optional[List[str]] = None,
        tool_args: Optional[List[str]] = None,
        tool_responses: Optional[List[str]] = None,
        iteration: int = 0,
        workspace_id: Optional[str] = None,
        save_reports: bool = True,
        *,
        _skip_init_report_file: bool = False,
    ):
        self._workspace_id = (
            workspace_id
            if workspace_id is not None
            else dt.now(BEIJING_TZ).strftime("%Y-%m-%d-%H-%M-%S")
        )
        self._save_reports = save_reports
        self._question = question
        self._report = report
        self._actions = list(actions) if actions is not None else None
        self._tool_args = list(tool_args) if tool_args is not None else None
        self._tool_responses = (
            list(tool_responses) if tool_responses is not None else None
        )
        self._iteration = iteration
        self._lock = threading.RLock()
        if self._save_reports and not _skip_init_report_file:
            self._init_report_file()

    @property
    def lock(self) -> threading.RLock:
        """Hold this lock when performing compound operations across multiple get/set."""
        return self._lock

    def _get_report_dir(self) -> str:
        """Directory for this workspace's logs (time-based folder)."""
        return os.path.abspath(f"workspace_log_{self._workspace_id}")

    def _get_question_reports_path(self) -> str:
        """Path to the .txt that records question and each round's report."""
        return os.path.join(
            self._get_report_dir(),
            "question_and_reports.txt",
        )

    def _get_tool_calling_path(self) -> str:
        """Path to the .txt that records each round's tool calling result."""
        return os.path.join(
            self._get_report_dir(),
            "tool_calling_results.txt",
        )

    def _init_report_file(self) -> None:
        """Create log folder and two .txt files when save_reports is True."""
        if not self._save_reports:
            return
        try:
            log_dir = self._get_report_dir()
            os.makedirs(log_dir, exist_ok=True)
            qr_path = self._get_question_reports_path()
            with open(qr_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Workspace workspace_id={self._workspace_id}\n"
                )
                f.write("Question and per-round reports below.\n\n")
            tc_path = self._get_tool_calling_path()
            with open(tc_path, "w", encoding="utf-8") as f:
                f.write(
                    f"Workspace workspace_id={self._workspace_id}\n"
                )
                f.write("Per-round tool calling results below.\n\n")
        except OSError:
            pass

    def _append_to_report_file(
        self,
        iteration: int,
        update_type: str,
        content: Optional[str],
    ) -> None:
        """Append to the appropriate log file; no-op if save_reports is False.

        - question and report -> question_and_reports.txt
        - tool_calling_result -> tool_calling_results.txt
        - Other update types are not written (only these two files are used).
        """
        if not self._save_reports:
            return
        try:
            if update_type in ("question", "report"):
                path = self._get_question_reports_path()
                with open(path, "a", encoding="utf-8") as f:
                    if update_type == "question":
                        f.write("--- Question ---\n")
                    else:
                        f.write(f"--- Report (iteration {iteration}) ---\n")
                    if content is not None:
                        f.write(str(content))
                        if content and not content.endswith("\n"):
                            f.write("\n")
                    f.write("\n")
            elif update_type == "tool_calling_result":
                path = self._get_tool_calling_path()
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"--- Tool calling (iteration {iteration}) ---\n")
                    if content is not None:
                        f.write(str(content))
                        if content and not content.endswith("\n"):
                            f.write("\n")
                    f.write("\n")
        except OSError:
            pass

    def _format_tool_calling_result(
        self,
        actions: List[str],
        tool_args: Optional[List[Any]],
        tool_responses: Optional[List[str]],
    ) -> str:
        """Format actions + args + responses so each tool call is grouped (arg + response)."""
        lines: List[str] = []
        args_list = tool_args or []
        resp_list = tool_responses or []
        n = len(actions)
        for i in range(n):
            name = actions[i] if i < len(actions) else ""
            arg = args_list[i] if i < len(args_list) else ""
            resp = resp_list[i] if i < len(resp_list) else ""
            if isinstance(arg, dict):
                arg_str = json.dumps(arg, ensure_ascii=False, indent=2)
            else:
                arg_str = str(arg) if arg else "{}"
            lines.append(f"--- Tool call {self.get_iteration()}-{i + 1}: {name} ({self.get_workspace_id()}) ---")
            lines.append("Args:")
            lines.append(arg_str)
            lines.append("Response:")
            lines.append(resp if resp else "(empty)")
            lines.append("")
        return "\n".join(lines)

    # --- Getters (thread-safe, return copies for list fields) ---

    def get_workspace_id(self) -> str:
        with self._lock:
            return self._workspace_id

    def get_save_reports(self) -> bool:
        with self._lock:
            return self._save_reports

    def get_question(self) -> Optional[str]:
        with self._lock:
            return self._question

    def get_report(self) -> Optional[str]:
        with self._lock:
            return self._report

    def get_actions(self) -> Optional[List[str]]:
        with self._lock:
            return list(self._actions) if self._actions is not None else None

    def get_tool_args(self) -> Optional[List[str]]:
        with self._lock:
            return list(self._tool_args) if self._tool_args is not None else None

    def get_tool_responses(self) -> Optional[List[str]]:
        with self._lock:
            return (
                list(self._tool_responses)
                if self._tool_responses is not None
                else None
            )

    def get_iteration(self) -> int:
        with self._lock:
            return self._iteration

    # --- Setters (thread-safe, store copies for list fields) ---

    def set_question(self, value: Optional[str]) -> None:
        with self._lock:
            self._question = value
            it = self._iteration
        self._append_to_report_file(it, "question", value)

    def set_report(self, value: Optional[str]) -> None:
        with self._lock:
            self._report = value
            it = self._iteration
        self._append_to_report_file(it, "report", value)

    def set_actions(self, value: Optional[List[str]]) -> None:
        with self._lock:
            self._actions = list(value) if value is not None else None
            it = self._iteration
        self._append_to_report_file(
            it, "actions", str(value) if value is not None else None
        )

    def set_tool_args(self, value: Optional[List[str]]) -> None:
        with self._lock:
            self._tool_args = list(value) if value is not None else None
            it = self._iteration
        self._append_to_report_file(
            it, "tool_args", str(value) if value is not None else None
        )

    def set_tool_responses(self, value: Optional[List[str]]) -> None:
        with self._lock:
            self._tool_responses = (
                list(value) if value is not None else None
            )
            it = self._iteration
        self._append_to_report_file(
            it, "tool_responses", str(value) if value is not None else None
        )

    def set_tool_calling_result(
        self,
        actions: List[str],
        tool_args: Optional[List[Union[str, dict]]] = None,
        tool_responses: Optional[List[str]] = None,
    ) -> None:
        """Set actions, tool_args, and tool_responses in one go, then append one
        beautified block to the report file (each tool call's arg and response
        grouped together).
        """
        args_raw = tool_args or []
        resp_list = list(tool_responses) if tool_responses else []
        # Store tool_args as list of JSON strings
        args_stored: List[str] = []
        for a in args_raw:
            if isinstance(a, dict):
                args_stored.append(json.dumps(a, ensure_ascii=False))
            else:
                args_stored.append(str(a) if a else "{}")
        with self._lock:
            self._actions = list(actions)
            self._tool_args = args_stored if args_stored else None
            self._tool_responses = resp_list if resp_list else None
            it = self._iteration
        content = self._format_tool_calling_result(
            actions, tool_args, tool_responses
        )
        self._append_to_report_file(it, "tool_calling_result", content)

    def set_iteration(self, value: int) -> None:
        with self._lock:
            self._iteration = value
        self._append_to_report_file(value, "iteration", str(value))

    def increment_iteration(self) -> int:
        """Increment iteration by 1 and return the new value."""
        with self._lock:
            self._iteration += 1
            it = self._iteration
        # self._append_to_report_file(it, "iteration", str(it))
        return it

    # --- Append helpers (thread-safe, for single-item appends) ---

    def append_action(self, tool_name: str) -> None:
        with self._lock:
            if self._actions is None:
                self._actions = []
            self._actions.append(tool_name)
            it = self._iteration
        self._append_to_report_file(it, "action", tool_name)

    def append_tool_arg(self, args_json: str) -> None:
        with self._lock:
            if self._tool_args is None:
                self._tool_args = []
            self._tool_args.append(args_json)
            it = self._iteration
        self._append_to_report_file(it, "tool_arg", args_json)

    def append_tool_response(self, response: str) -> None:
        with self._lock:
            if self._tool_responses is None:
                self._tool_responses = []
            self._tool_responses.append(response)
            it = self._iteration
        self._append_to_report_file(it, "tool_response", response)

    def to_messages(
        self,
        system_prompt: Optional[str] = None,
    ) -> List[RoleMessage]:
        """Convert this Workspace to a list of RoleMessage (system/user/assistant/tool).

        Role mapping:
        - question -> one UserMessage.
        - report -> one AssistantMessage (evolving report from previous round).
        - actions + tool_args + tool_responses -> one AssistantMessage with
          tool_calls, then one ToolMessage per pair.

        Each message's content is prefixed with a short hint indicating which part
        of the current iteration's workspace it comes from. The worker converts
        these to the API format via message.to_dict().
        """
        messages: List[RoleMessage] = []

        if system_prompt is not None:
            messages.append(SystemMessage(system_prompt))

        question = self.get_question()
        if not question:
            return messages

        messages.append(
            UserMessage("[Workspace: research question (current iteration)]\n\n" + question),
        )

        report = self.get_report()
        if report:
            messages.append(
                AssistantMessage(
                    "[Workspace: evolving report (current iteration)]\n\n" + report,
                ),
            )

        act_list = self.get_actions() or []
        args_list = self.get_tool_args() or []
        resp_list = self.get_tool_responses() or []

        if act_list and resp_list and len(act_list) == len(resp_list):
            tool_calls = []
            for idx, (name, _) in enumerate(zip(act_list, resp_list)):
                tool_call_id = f"{WORKSPACE_TOOL_CALL_ID}-{idx}"
                args_str = args_list[idx] if idx < len(args_list) else "{}"
                tool_calls.append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                })
            messages.append(
                AssistantMessage(
                    "[Workspace: previous assistant tool calls (current iteration); tool results follow in tool messages.]",
                    tool_calls=tool_calls,
                ),
            )

            for idx, (name, content) in enumerate(zip(act_list, resp_list)):
                tool_call_id = f"{WORKSPACE_TOOL_CALL_ID}-{idx}"
                messages.append(
                    ToolMessage(
                        content=f"[Workspace: tool response for {name} (current iteration)]\n\n{content}",
                        tool_call_id=tool_call_id,
                        name=name,
                    ),
                )

        return messages
