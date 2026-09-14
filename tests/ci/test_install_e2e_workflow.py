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


def _init_repo(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Install E2E test")
    _git(path, "config", "user.email", "install-e2e-test@invalid")


def _commit_installer(repo: Path, body: str, message: str) -> str:
    installer = repo / "scripts/install.sh"
    installer.parent.mkdir(exist_ok=True)
    installer.write_text(f"#!/bin/sh\n# {body}\nexit 0\n", encoding="utf-8")
    _git(repo, "add", "scripts/install.sh")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD^{commit}")


def test_install_e2e_uses_upstream_checkout_without_leaking_credential_or_source_state():
    """Matrix legs consume an upstream checkout outside the source and token scope."""
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
    move_step = _one_step(
        steps,
        lambda step: step.get("name") == "Move upstream checkout outside source",
        "upstream move step",
    )
    run_step = _one_step(
        steps,
        lambda step: step.get("name") == "Run install + update E2E",
        "install E2E step",
    )

    assert source_checkout["with"]["fetch-depth"] == 0
    upstream_with = upstream_checkout["with"]
    assert upstream_with["repository"] == "NousResearch/hermes-agent"
    assert upstream_with["ref"] == "${{ inputs.install-ref }}"
    assert upstream_with["fetch-depth"] == 0
    assert upstream_with["path"] == ".install-upstream"
    assert upstream_with["persist-credentials"] is False
    assert move_step["env"]["UPSTREAM_CHECKOUT"] == (
        "${{ runner.temp }}/install-upstream"
    )
    assert move_step["run"] == 'mv .install-upstream "$UPSTREAM_CHECKOUT"'
    assert run_step["env"]["HERMES_DEV_SANDBOX_UPSTREAM"] == (
        "${{ runner.temp }}/install-upstream"
    )


@pytest.mark.parametrize("install_ref", ["v1", "refs/heads/main"])
def test_dev_sandbox_resolves_distinct_upstream_tag_and_branch_without_mutating_source(
    tmp_path: Path,
    install_ref: str,
):
    """The real resolver uses its upstream checkout, not same-named source refs."""
    source = tmp_path / "source"
    _init_repo(source)
    _commit_installer(source, "source tag decoy", "source tag decoy")
    _git(source, "tag", "v1")
    _commit_installer(source, "source branch decoy", "source branch decoy")

    upstream_origin = tmp_path / "upstream-origin"
    _init_repo(upstream_origin)
    upstream_tag = _commit_installer(upstream_origin, "upstream tag", "upstream tag")
    _git(upstream_origin, "tag", "v1")
    upstream_branch = _commit_installer(
        upstream_origin,
        "upstream branch",
        "upstream branch",
    )

    upstream_checkout = tmp_path / "upstream-checkout"
    upstream_checkout.mkdir()
    _git(upstream_checkout, "init", "-q")
    _git(upstream_checkout, "remote", "add", "origin", str(upstream_origin))
    _git(
        upstream_checkout,
        "fetch",
        "-q",
        "origin",
        "+refs/heads/*:refs/remotes/origin/*",
        "+refs/tags/*:refs/tags/*",
    )
    if install_ref == "v1":
        _git(upstream_checkout, "checkout", "-q", "--detach", "refs/tags/v1")
        expected_release = upstream_tag
    else:
        _git(
            upstream_checkout,
            "checkout",
            "-q",
            "-B",
            "main",
            "refs/remotes/origin/main",
        )
        expected_release = upstream_branch

    assert _git(source, "rev-parse", f"{install_ref}^{{commit}}") != expected_release
    refs_before = _git(source, "for-each-ref", "--format=%(refname) %(objectname)")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_unshare = fake_bin / "unshare"
    fake_unshare.write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
    fake_unshare.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()

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
        env={
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(home),
            "HERMES_SANDBOX_SOURCE_ROOT": str(source),
            "HERMES_DEV_SANDBOX_UPSTREAM": str(upstream_checkout),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 73, completed.stderr
    assert (
        f"[sandbox] fake main: upstream {install_ref} ({expected_release})"
        in completed.stderr
    )
    assert "could not resolve upstream ref" not in completed.stderr
    assert (
        _git(source, "for-each-ref", "--format=%(refname) %(objectname)") == refs_before
    )
