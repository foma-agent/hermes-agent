"""Literal wrapper argv grammar shared by command-position and policy detection."""

_COMMAND_WRAPPER_WORDS = {"sudo", "env", "exec", "nohup", "setsid", "time", "command", "builtin",
                          "nice", "timeout", "stdbuf", "ionice", "chrt", "taskset", "chroot", "xargs",
                          "uv", "poetry", "pipx"}
# These package runners execute argv only through their ``run`` subcommand. Treating every
# subcommand as a wrapper would turn data following e.g. ``uv sync`` into an executable.
_COMMAND_WRAPPER_REQUIRED_POSITIONAL = {"uv": "run", "poetry": "run", "pipx": "run"}
_SUDO_OPTIONS_WITH_ARG = {"-c", "--close-from", "-g", "--group", "-h", "--host", "-p", "--prompt", "-u", "--user"}
# Adapted from embwl0x's command-position work in #76063. Option operands are
# data, not executable positions; option spelling remains case-sensitive.
_COMMAND_WRAPPER_OPTIONS_WITH_ARG = {
    "chroot": {"--groups", "--userspec"},
    "sudo": _SUDO_OPTIONS_WITH_ARG,
    "env": {"-a", "--argv0", "-C", "--chdir", "-S", "--split-string", "-u", "--unset"},
    "exec": {"-a"}, "nice": {"-n", "--adjustment"},
    "time": {"-f", "--format", "-o", "--output"},
    "timeout": {"-k", "--kill-after", "-s", "--signal"},
    "stdbuf": {"-e", "--error", "-i", "--input", "-o", "--output"},
    "ionice": {"-c", "--class", "-n", "--classdata"},
    "xargs": {"-a", "--arg-file", "-d", "--delimiter", "-E", "-I", "-J", "-L", "-n",
              "--max-args", "-P", "--max-procs", "-R", "-S", "-s", "--max-chars",
              "--process-slot-var"},
    "uv": {"--directory", "--project", "--config-file", "--python", "--with", "--with-editable",
           "--with-requirements", "--env-file", "--package"},
    "poetry": {"-C", "--directory", "-P", "--project"},
    "pipx": {"--index-url", "--pip-args", "--python", "--spec", "--suffix"},
}
_COMMAND_WRAPPER_NON_EXECUTING_OPTIONS = {
    "command": {"-v", "-V"}, "chrt": {"-p", "--pid"},
    "ionice": {"-p", "--pid", "--pgid", "--uid"}, "taskset": {"-p", "--pid"},
}
_XARGS_SHORT_OPTIONS_WITH_ARG = frozenset(
    option[1]
    for option in _COMMAND_WRAPPER_OPTIONS_WITH_ARG["xargs"]
    if option.startswith("-") and not option.startswith("--") and len(option) == 2
)
_XARGS_SHORT_OPTIONS_WITH_OPTIONAL_ARG = frozenset("eil")
_COMMAND_WRAPPER_POSITIONAL_ARGS = {"chroot": 1, "chrt": 1, "taskset": 1, "timeout": 1}


def _wrapper_option_consumes_next_arg(wrapper: str, token: str) -> bool:
    """Whether a wrapper option owns the following argv word."""
    option = token.split("=", 1)[0]
    if "=" in token:
        return False
    if option in _COMMAND_WRAPPER_OPTIONS_WITH_ARG.get(wrapper, set()):
        return True
    if wrapper != "xargs" or option.startswith("--"):
        return False
    # GNU/BSD xargs accept clustered short options. The first option with an
    # argument owns the rest of its token; only a required owner at the end
    # consumes the next word. Deprecated -e/-i/-l take optional attached-only
    # arguments, so they stop the cluster without swallowing the command.
    chars = option[1:]
    for index, char in enumerate(chars):
        if char in _XARGS_SHORT_OPTIONS_WITH_OPTIONAL_ARG:
            return False
        if char in _XARGS_SHORT_OPTIONS_WITH_ARG:
            return index == len(chars) - 1
    return False
