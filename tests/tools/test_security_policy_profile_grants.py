"""Merge witness for #59337 one-shot consent plus upstream profile grant isolation.

Ordinary grant isolation and retirement are retained upstream controls, not new
security boundaries introduced by the policy-mutation guard.
"""

from pathlib import Path

import pytest


@pytest.fixture
def profile_grants(tmp_path, monkeypatch):
    from tools import approval

    launch = tmp_path / ".hermes"
    profiles = (launch / "profiles" / "a", launch / "profiles" / "b")
    for home in (launch, *profiles):
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("command_allowlist: []\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    grants = {}
    monkeypatch.setattr(approval, "_session_approved", grants)
    monkeypatch.setattr(approval, "_permanent_approved", set())
    monkeypatch.setattr(approval, "_permanent_approved_by_home", {})
    monkeypatch.setattr(approval, "_permanent_baseline_by_home", {})
    return profiles, grants


def test_policy_mutation_cannot_be_stored_in_routed_session(profile_grants):
    from tools import approval
    from tools.delegate_tool_routes import delegated_profile_scope

    (profile_a, _), grants = profile_grants
    session = "policy-profile-merge-witness"
    ordinary = "world/other-writable permissions"
    policy = approval._SECURITY_CONFIG_APPROVAL_KEY
    try:
        with delegated_profile_scope(profile_a):
            approval.approve_session(session, ordinary)
            assert approval.is_approved(session, ordinary)
            approval.approve_session(session, policy)
            assert not approval.is_approved(session, policy)
            # Reading denies policy keys independently; also prove storage rejected it.
            assert all(policy not in patterns for patterns in grants.values())
    finally:
        approval.clear_session(session)


def test_retained_ordinary_grants_isolate_profiles_and_clear_together(profile_grants):
    from tools import approval
    from tools.delegate_tool_routes import delegated_profile_scope

    (profile_a, profile_b), grants = profile_grants
    session = "ordinary-profile-retained-control"
    launch_key, a_key, b_key = "launch-control", "a-control", "b-control"
    try:
        approval.approve_session(session, launch_key)
        for home, own_key, other_key in (
            (profile_a, a_key, b_key), (profile_b, b_key, a_key),
        ):
            with delegated_profile_scope(home):
                assert not approval.is_approved(session, launch_key)
                assert not approval.is_approved(session, other_key)
                approval.approve_session(session, own_key)
                assert approval.is_approved(session, own_key)
        assert approval.is_approved(session, launch_key)
        assert not approval.is_approved(session, a_key)
        assert not approval.is_approved(session, b_key)
        with delegated_profile_scope(profile_a):
            assert approval.is_approved(session, a_key)
            assert not approval.is_approved(session, b_key)
        approval.clear_session(session)
        assert not approval.is_approved(session, launch_key)
        for home, own_key in ((profile_a, a_key), (profile_b, b_key)):
            with delegated_profile_scope(home):
                assert not approval.is_approved(session, own_key)
        assert not grants
    finally:
        approval.clear_session(session)
