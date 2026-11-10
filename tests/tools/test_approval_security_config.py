"""Carrier policy CLI behavior contracts for PR #59337 / issue #59293."""
import pytest

import tools.approval as approval_module
from tools.approval import approve_session, detect_dangerous_command, is_approved, load_permanent


class TestSecurityConfigCLI:
    def test_policy_mutation_cli_requires_approval(self):
        dangerous, key, desc = detect_dangerous_command(
            "hermes config set approvals.mode off"
        )

        assert dangerous is True
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize("action", ["set", "unset"])
    @pytest.mark.parametrize(
        "key",
        [
            "approvals.mode",
            "security.tirith_enabled",
            "command_allowlist",
            "yolo",
        ],
    )
    def test_policy_mutation_cli_protected_keys(self, action, key):
        value = " false" if action == "set" else ""
        dangerous, pattern_key, desc = detect_dangerous_command(
            f"hermes config {action} {key}{value}"
        )

        assert dangerous is True
        assert pattern_key == "modify Hermes security policy via config"
        assert desc == pattern_key

    @pytest.mark.parametrize(
        "command",
        [
            "hermes --profile prod config unset security.tirith_enabled",
            "hermes -p prod config set command_allowlist '[\"git status\"]'",
            "hermes --reasoning high config set approvals.mode off",
            "hermes config --profile=prod set --force approvals.mode off",
            "hermes config set -- approvals.mode off",
            "hermes config set 'approvals'.mode off",
            'hermes config set ap"provals".mode off',
            'hermes config set approvals".mode" off',
        ],
    )
    def test_policy_mutation_cli_flags_and_quoting(self, command):
        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize(
        "command",
        [
            "uv run hermes config set approvals.mode off",
            "uv run -p 3.12 hermes config set approvals.mode off",
            "uv --cache-dir /tmp/c run hermes config set approvals.mode off",
            "uv --color always run hermes config unset security.tirith_enabled",
            "uv --directory /tmp run --no-project hermes config unset security.tirith_enabled",
            "uv tool run --from . hermes config unset security.tirith_enabled",
            "uvx --from . hermes config set approvals.mode off",
            "uv run python -m hermes_cli.main config set yolo true",
            "poetry run hermes config set command_allowlist '[]'",
            "pipx run hermes config set yolo true",
            "xargs hermes config unset security.tirith_enabled",
            "xargs -n 1 hermes config set approvals.mode off",
            "xargs -J % hermes config unset security.tirith_enabled",
        ],
    )
    def test_policy_mutation_cli_runner_entrypoints(self, command):
        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize("option", ["-i", "-l", "-e", "--replace", "--eof", "--max-lines"])
    def test_policy_mutation_xargs_optional_argument_options(self, option):
        command = f"xargs {option} hermes config set approvals.mode off"

        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize(
        "command",
        [
            "xargs -tn 1 hermes config set approvals.mode off",
            "xargs -tI {} hermes config unset security.tirith_enabled",
            "xargs -0P 4 hermes config set command_allowlist []",
        ],
    )
    def test_policy_mutation_xargs_clustered_argument_options(self, command):
        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    def test_policy_mutation_eval_payload_keeps_one_shot_key(self):
        command = 'eval "hermes config set command_allowlist []"'

        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize(
        "command",
        [
            "sh -c 'hermes config set approvals.mode off' ; git push --force",
            r".\hermes.exe config unset security.tirith_enabled && echo x >> ~/.bashrc",
        ],
    )
    def test_policy_mutation_key_wins_across_detection_variants(self, command):
        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize(
        "command",
        [
            r".\hermes.exe config set approvals.mode off",
            r"venv\Scripts\hermes.exe config set approvals.mode off",
            r".\uv.exe run hermes config unset security.tirith_enabled",
            r".\poetry.exe run hermes config set command_allowlist []",
        ],
    )
    def test_policy_mutation_windows_relative_entrypoints(self, command):
        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize(
        "command",
        [
            "python -m hermes_cli.main config set approvals.mode off",
            "python -mhermes_cli.main config unset security.tirith_enabled",
            "python -u -m hermes_cli.main config set yolo true",
            "python -Im hermes_cli.main config set command_allowlist '[]'",
            "python hermes_cli/main.py config set approvals.mode off",
            "/usr/bin/python ./hermes_cli/main.py config unset security.tirith_enabled",
        ],
    )
    def test_policy_mutation_python_entrypoints(self, command):
        dangerous, key, desc = detect_dangerous_command(command)

        assert dangerous is True, command
        assert key == "modify Hermes security policy via config"
        assert desc == key

    @pytest.mark.parametrize(
        "command",
        [
            "hermes config get approvals.mode",
            "hermes config get display.show_cost",
            "hermes config set display.show_cost true",
            "hermes config unset display.show_cost",
            "python -m hermes_cli.main config set display.show_cost true",
            "uv sync hermes config set approvals.mode off",
            "xargs echo hermes config set approvals.mode off",
        ],
    )
    def test_ordinary_config_cli_operations_stay_safe(self, command):
        assert detect_dangerous_command(command) == (False, None, None)

    def test_policy_mutation_ignores_session_permanent_and_preloaded_approvals(self):
        _, key, _ = detect_dangerous_command("hermes config set approvals.mode off")
        session_key = "policy-mutation-preloaded"
        previous_session = approval_module._session_approved.get(session_key)
        governing = approval_module._permanent_set()
        previous_permanent = set(governing)

        try:
            approve_session(session_key, key)
            approval_module.approve_permanent(key)
            load_permanent({key})
            assert key not in approval_module._session_approved.get(session_key, set())
            assert key not in governing

            # Older processes or fixtures may already contain the key; reads still fail closed.
            approval_module._session_approved[session_key] = {key}
            governing.add(key)
            assert is_approved(session_key, key) is False
        finally:
            if previous_session is None:
                approval_module._session_approved.pop(session_key, None)
            else:
                approval_module._session_approved[session_key] = previous_session
            governing.clear()
            governing.update(previous_permanent)

    def test_routed_profile_drops_preloaded_policy_mutation_key(self, monkeypatch, tmp_path):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        _, key, _ = detect_dangerous_command("hermes config set approvals.mode off")
        previous_profiles = {
            home: set(patterns)
            for home, patterns in approval_module._permanent_approved_by_home.items()
        }
        monkeypatch.setattr(approval_module, "_read_permanent_allowlist", lambda: {key})
        token = set_hermes_home_override(tmp_path / "profile")

        try:
            approval_module._permanent_approved_by_home.clear()
            assert approval_module._is_permanently_approved(key) is False
            assert key not in approval_module._permanent_set()
        finally:
            reset_hermes_home_override(token)
            approval_module._permanent_approved_by_home.clear()
            approval_module._permanent_approved_by_home.update(previous_profiles)
