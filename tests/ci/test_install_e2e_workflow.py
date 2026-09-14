from pathlib import Path
import subprocess

import pytest
import yaml


_REPO = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO / ".github/workflows/install-e2e-run.yml"
_SANDBOX = _REPO / "scripts/dev-sandbox.sh"


def _steps() -> list[dict]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["e2e"]["steps"]


def _one_step(steps: list[dict], predicate, description: str) -> dict:
    matches = [step for step in steps if predicate(step)]
    assert len(matches) == 1, f"expected one {description}, found {len(matches)}"
    return matches[0]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def test_install_e2e_uses_authenticated_upstream_checkout_for_release_refs():
    """Matrix legs resolve their input from upstream without anonymous fetches."""
    steps = _steps()
    source_checkout = _one_step(
        steps,
        lambda step: (
            str(step.get("uses", "")).startswith("actions/checkout@")
            and "repository" not in step.get("with", {})
        ),
        "source checkout step",
    )
    upstream_checkout = _one_step(
        steps,
        lambda step: (
            str(step.get("uses", "")).startswith("actions/checkout@")
            and step.get("with", {}).get("repository") == "NousResearch/hermes-agent"
        ),
        "upstream checkout step",
    )
    exclude_step = _one_step(
        steps,
        lambda step: (
            step.get("name") == "Exclude upstream checkout from source snapshot"
        ),
        "upstream exclusion step",
    )
    run_step = _one_step(
        steps,
        lambda step: step.get("name") == "Run install + update E2E",
        "install E2E step",
    )

    assert source_checkout["with"]["fetch-depth"] == 0
    assert upstream_checkout["with"] == {
        "repository": "NousResearch/hermes-agent",
        "ref": "${{ inputs.install-ref }}",
        "fetch-depth": 0,
        "path": ".install-upstream",
    }
    assert "${{ github.token }}" not in str(upstream_checkout)
    assert "/.install-upstream/" in exclude_step["run"]
    assert run_step["env"]["HERMES_DEV_SANDBOX_UPSTREAM"] == (
        "${{ github.workspace }}/.install-upstream"
    )


@pytest.mark.parametrize("install_ref", ["v1", "refs/heads/main"])
def test_dev_sandbox_resolves_local_tag_and_branch_without_mutating_source(
    tmp_path: Path,
    install_ref: str,
):
    """The workflow override reaches the real ref fetch without touching source."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "main")
    _git(source, "config", "user.name", "Install E2E test")
    _git(source, "config", "user.email", "install-e2e-test@invalid")
    installer = source / "scripts/install.sh"
    installer.parent.mkdir()
    installer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    _git(source, "add", "scripts/install.sh")
    _git(source, "commit", "-q", "-m", "release")
    _git(source, "tag", "v1")
    refs_before = _git(source, "for-each-ref", "--format=%(refname) %(objectname)")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_unshare = fake_bin / "unshare"
    fake_unshare.write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
    fake_unshare.chmod(0o755)

    env = {
        "PATH": f"{fake_bin}:{Path('/usr/bin')}:/bin",
        "HOME": str(tmp_path / "home"),
        "HERMES_SANDBOX_SOURCE_ROOT": str(source),
        "HERMES_DEV_SANDBOX_UPSTREAM": str(source),
    }
    completed = subprocess.run(
        [
            str(_SANDBOX),
            "install",
            "--root",
            "--install-ref",
            install_ref,
            "--",
            "--skip-setup",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 73, completed.stderr
    release = _git(source, "rev-parse", "HEAD^{commit}")
    assert (
        f"[sandbox] fake main: upstream {install_ref} ({release})" in completed.stderr
    )
    assert "could not resolve upstream ref" not in completed.stderr
    assert (
        _git(source, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    )
