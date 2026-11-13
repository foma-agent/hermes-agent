"""Structural Hermes CLI policy-write and literal child-bypass detection."""
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
        # The editor can replace any policy namespace, regardless of its initial contents.
        if without_profiles[index + 1] == "edit":
            return True
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


def _child_yolo_argv(argv: list[str]) -> bool:
    """Recognize a literal flag, respecting CLI option values and the data boundary."""
    from hermes_cli._parser import top_level_value_flag_sets

    required, optional = top_level_value_flag_sets()
    chat_values = (required - {"-z", "--oneshot"}) | {
        "-q", "--query", "--query-file", "--image", "--base-url", "--api-key",
        "--format", "--max-turns", "--run-budget", "--source",
    }
    index, chat = 0, False
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return False
        if token == "--yolo":
            return True
        if not token.startswith("-"):
            if token != "chat" or chat:
                return False
            chat = True
            index += 1
            continue
        option, equals, _ = token.partition("=")
        values = chat_values if chat else required
        if not equals and (
            option in values | {"--profile", "-p"}
            or (option in optional and index + 1 < len(argv) and not argv[index + 1].startswith("-"))
        ):
            index += 2
        else:
            # Short value options also accept attached values, e.g. -q--yolo.
            index += 1
    return False


def _literal_env_yolo(tokens: list[str], inherited: bool) -> bool:
    """Apply the existing env wrapper's literal options/assignments, without expansion."""
    from tools.approval_detection import _wrapper_option_consumes_next_arg
    from utils import is_truthy_value

    index, value, options = 1, inherited, True
    while index < len(tokens):
        token = tokens[index]
        if options and token == "--":
            options = False
        elif options and token in {"-i", "--ignore-environment", "-"}:
            value = False
        elif options and (token in {"-u", "--unset"} or token.startswith("--unset=") or token.startswith("-u")):
            name = token.split("=", 1)[1] if "=" in token else token[2:]
            if token in {"-u", "--unset"}:
                index += 1
                name = tokens[index] if index < len(tokens) else ""
            if name == "HERMES_YOLO_MODE":
                value = False
        elif options and token.startswith("-"):
            if token.startswith("-S") or token.split("=", 1)[0] == "--split-string":
                break  # _env_split_payload projects this separately as literal argv.
            index += int(_wrapper_option_consumes_next_arg("env", token))
        elif "=" in token:
            name, assignment = token.split("=", 1)
            if name == "HERMES_YOLO_MODE":
                value = is_truthy_value(assignment)
        else:
            break
        index += 1
    return value


def _literal_yolo_before(source: str, word_start: int, inherited: bool, starts: list[int]) -> bool:
    """Recover assignments discarded by the command-word scanner, scoped to this command.

    Only assignment positions and actual env wrappers carry authority. Quoted
    wrapper operands and arbitrary argv data must never become environment state.
    """
    from bisect import bisect_right
    from tools.approval_detection import (
        _ENV_ASSIGNMENT_RE, _iter_shell_command_word_spans,
        _shell_segment_tokens, _shell_tokens_with_spans,
    )
    from utils import is_truthy_value

    start = starts[bisect_right(starts, word_start) - 1]
    prefix = source[start:word_start]
    value = inherited
    for token, begin, end, _ in _shell_tokens_with_spans(prefix, 0) or []:
        if not _ENV_ASSIGNMENT_RE.fullmatch(prefix[begin:end]):
            break
        name, assignment = token.split("=", 1)
        if name == "HERMES_YOLO_MODE":
            value = is_truthy_value(assignment)
    for begin, _, _ in _iter_shell_command_word_spans(prefix):
        tokens = _shell_segment_tokens(prefix, begin)
        if tokens and os.path.basename(tokens[0]).lower() == "env":
            value = _literal_env_yolo(tokens, value)
    return value


def _security_config_entrypoint_mutation(
    tokens: list[str], *, launch_yolo: bool = False, config_only: bool = False,
) -> bool:
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
    # Reuse the legacy "modify Hermes security policy via config" identity:
    # child bypass launches need the same one-operation, no-cache consent.
    return (not config_only and (launch_yolo or _child_yolo_argv(argv))) or _security_config_argv_mutation(argv)


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
    """Find policy writes or literal child bypasses under the legacy config identity.

    This is an accident-prevention front door, not same-UID containment: dynamic
    scripts, expanded environment values and unrecognized runners are not resolved.
    """
    from tools.approval_detection import (
        _env_split_payload, _iter_shell_command_starts, _iter_shell_command_word_spans,
        _shell_command_segment, _shell_segment_tokens,
    )

    pending, seen = [(command, False)], set()
    while pending:
        source, inherited = pending.pop()
        if (source, inherited) in seen:
            continue
        seen.add((source, inherited))
        # Shell starts belong to the source, not each command word. Recover them
        # once, and skip environment work entirely when no literal value exists.
        starts = sorted(_iter_shell_command_starts(source)) if inherited or "HERMES_YOLO_MODE" in source else []
        for word_start, _, _ in _iter_shell_command_word_spans(source):
            tokens = _shell_segment_tokens(_shell_command_segment(source, word_start), 0)
            if not tokens:
                continue
            launch_yolo = bool(starts) and _literal_yolo_before(source, word_start, inherited, starts)
            if _security_config_entrypoint_mutation(tokens, launch_yolo=launch_yolo):
                return True
            if os.path.basename(tokens[0]).lower() == "env":
                # Reuse the bounded literal env grammar and quote argv on projection:
                # split-string data is not shell code, and the suffix stays argv.
                payload = _env_split_payload(tokens)
                if payload:
                    pending.append((payload, _literal_env_yolo(tokens, launch_yolo)))
            # Preserve the conservative historical config scan. Child flags use
            # only actual command positions above, never runner argument data.
            payload_start = _security_config_runner_payload_start(tokens)
            if payload_start is not None and any(
                _security_config_entrypoint_mutation(tokens[index:], config_only=True)
                for index in range(payload_start, len(tokens))
            ):
                return True
            # eval concatenates its argv with spaces and parses the result as shell code.
            # Re-scan that bounded, strictly shorter payload instead of treating a quoted
            # policy mutation as inert argument data.
            if os.path.basename(tokens[0]).lower() == "eval" and len(tokens) > 1:
                payload = " ".join(tokens[1:])
                if (payload, launch_yolo) not in seen:
                    pending.append((payload, launch_yolo))
    return False
