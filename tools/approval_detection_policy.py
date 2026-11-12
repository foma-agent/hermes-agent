"""Structural Hermes CLI security-policy mutation detection."""
import os


def _security_config_argv_mutation(argv: list[str]) -> bool:
    """Whether dequoted Hermes argv writes an approval/security config namespace."""
    without_profiles, index = [], 0
    while index < len(argv):
        token = argv[index]
        if token in {"--profile", "-p"}:
            index += 2
            continue
        if token.startswith("--profile="):
            index += 1
            continue
        without_profiles.append(token)
        index += 1

    for index, token in enumerate(without_profiles):
        if token != "config" or index + 1 >= len(without_profiles):
            continue
        if without_profiles[index + 1] not in {"set", "unset"}:
            continue
        key_index = index + 2
        while key_index < len(without_profiles) and without_profiles[key_index] in {"--", "--force"}:
            key_index += 1
        if key_index >= len(without_profiles):
            continue
        key = without_profiles[key_index]
        # Replacing/removing the parent also changes the smart reviewer's authority.
        if key in {"yolo", "auxiliary"} or any(
            key == namespace or key.startswith(namespace + ".")
            for namespace in ("approvals", "security", "command_allowlist", "auxiliary.approval")
        ):
            return True
    return False


def _python_hermes_module_argv(tokens: list[str]) -> list[str] | None:
    """Return argv after a supported Python Hermes entrypoint, else None."""
    from tools.approval_detection import _INTERPRETER_WITH_ARG

    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token == "-m":
            if index + 1 < len(tokens) and tokens[index + 1] == "hermes_cli.main":
                return tokens[index + 2:]
            return None
        if token.startswith("-m") and token[2:] == "hermes_cli.main":
            return tokens[index + 1:]
        if token.startswith("-") and not token.startswith("--"):
            short_flags = token[1:]
            module_flag = short_flags.find("m")
            if module_flag >= 0 and not any(flag in "cWX" for flag in short_flags[:module_flag]):
                attached_module = short_flags[module_flag + 1:]
                if attached_module:
                    return tokens[index + 1:] if attached_module == "hermes_cli.main" else None
                if index + 1 < len(tokens) and tokens[index + 1] == "hermes_cli.main":
                    return tokens[index + 2:]
                return None
        if token == "--":
            return None
        if not token.startswith("-"):
            script = token.replace("\\", "/")
            is_hermes = script.endswith("hermes_cli/main.py") or script.rsplit("/", 1)[-1] == "hermes"
            return tokens[index + 1:] if is_hermes else None
        option = token.split("=", 1)[0]
        if "=" not in token and option in _INTERPRETER_WITH_ARG["python"]:
            index += 2
        else:
            index += 1
    return None


def _security_config_entrypoint_mutation(tokens: list[str]) -> bool:
    """Whether tokens begin with a Hermes CLI entrypoint that mutates policy."""
    from tools.approval_detection import _interpreter_family

    if not tokens:
        return False
    if os.path.basename(tokens[0]).lower() in {"hermes", "hermes.exe"}:
        argv = tokens[1:]
    elif _interpreter_family(tokens[0]) == "python":
        argv = _python_hermes_module_argv(tokens)
        if argv is None:
            return False
    else:
        return False
    return _security_config_argv_mutation(argv)


def _security_config_runner_payload_start(tokens: list[str]) -> int | None:
    """Locate argv payloads for runners whose option grammars change independently.

    Security-policy detection scans the payload instead of trusting an inevitably
    incomplete table of third-party options. That can prompt on ambiguous runner
    argv, but it cannot silently lose the real Hermes entrypoint.
    """
    if not tokens:
        return None
    runner = os.path.basename(tokens[0]).lower()
    if runner in {"uvx", "uvx.exe"}:
        return 1
    if runner in {"uv", "uv.exe"}:
        for index, token in enumerate(tokens[1:], 1):
            if token == "run":
                return index + 1
            if token == "tool" and index + 1 < len(tokens) and tokens[index + 1] in {"run", "uvx"}:
                return index + 2
        return None
    if runner in {"poetry", "poetry.exe", "pipx", "pipx.exe", "pdm", "pdm.exe",
                  "hatch", "hatch.exe", "rye", "rye.exe", "pipenv", "pipenv.exe"}:
        try:
            return tokens.index("run", 1) + 1
        except ValueError:
            return None
    return None


def _is_hermes_security_config_mutation(command: str) -> bool:
    """Parse command-position argv so shell quoting cannot hide a config mutation."""
    from tools.approval_detection import (
        _env_split_payload, _iter_shell_command_word_spans,
        _shell_command_segment, _shell_segment_tokens,
    )

    pending, seen = [command], set()
    while pending:
        source = pending.pop()
        if source in seen:
            continue
        seen.add(source)
        for word_start, _, _ in _iter_shell_command_word_spans(source):
            tokens = _shell_segment_tokens(_shell_command_segment(source, word_start), 0)
            if not tokens:
                continue
            if _security_config_entrypoint_mutation(tokens):
                return True
            if os.path.basename(tokens[0]).lower() == "env":
                # Reuse the bounded literal env grammar and quote argv on projection:
                # split-string data is not shell code, and the suffix stays argv.
                payload = _env_split_payload(tokens)
                if payload:
                    pending.append(payload)
            payload_start = _security_config_runner_payload_start(tokens)
            if payload_start is not None and any(
                _security_config_entrypoint_mutation(tokens[index:])
                for index in range(payload_start, len(tokens))
            ):
                return True
            # eval concatenates its argv with spaces and parses the result as shell code.
            # Re-scan that bounded, strictly shorter payload instead of treating a quoted
            # policy mutation as inert argument data.
            if os.path.basename(tokens[0]).lower() == "eval" and len(tokens) > 1:
                payload = " ".join(tokens[1:])
                if payload not in seen:
                    pending.append(payload)
    return False
