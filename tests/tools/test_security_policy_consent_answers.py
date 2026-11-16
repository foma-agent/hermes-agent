"""Gateway consent integration for the nonpersistable policy-mutation key."""

import pytest

from tools import approval


@pytest.fixture
def gateway_policy(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("approvals:\n  mode: manual\ncommand_allowlist: []\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "policy-consent-answers")
    monkeypatch.delenv("HERMES_SINGLE_QUERY_SESSION", raising=False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_timeout", lambda: 2)
    session = approval.get_current_session_key()
    cards = []

    def answer(choice):
        def notify(card):
            cards.append(card)
            assert approval.resolve_gateway_approval(
                session, choice, request_id=card["request_id"]
            ) == 1

        approval.register_gateway_notify(session, notify)
        result = approval.check_all_command_guards(
            "hermes config set approvals.mode off", "local"
        )
        policy = approval._SECURITY_CONFIG_APPROVAL_KEY
        assert all(policy not in keys for keys in approval._session_approved.values())
        assert policy not in approval._permanent_approved
        assert all(policy not in keys for keys in approval._permanent_approved_by_home.values())
        return result

    try:
        yield answer, cards, session, tmp_path
    finally:
        approval.unregister_gateway_notify(session)
        approval.clear_session(session)


def test_unknown_gateway_policy_answer_denies_without_storage(gateway_policy):
    answer, cards, session, home = gateway_policy
    assert answer("yes")["approved"] is False
    assert len(cards) == 1
    assert cards[0]["pattern_key"] == approval._SECURITY_CONFIG_APPROVAL_KEY
    assert cards[0]["allow_session"] is False
    assert cards[0]["allow_permanent"] is False
    assert not approval.is_approved(session, approval._SECURITY_CONFIG_APPROVAL_KEY)
    assert (home / "config.yaml").read_text() == "approvals:\n  mode: manual\ncommand_allowlist: []\n"


@pytest.mark.parametrize("choice", ["Approve", "session", "always"])
def test_policy_consent_is_once_and_reprompts(gateway_policy, choice):
    """Retained one-shot controls also apply to upstream's recognized alias."""
    answer, cards, session, home = gateway_policy
    original = (home / "config.yaml").read_text()
    assert answer(choice)["approved"] is True
    assert not approval.is_approved(session, approval._SECURITY_CONFIG_APPROVAL_KEY)
    assert not approval._is_permanently_approved(approval._SECURITY_CONFIG_APPROVAL_KEY)
    assert (home / "config.yaml").read_text() == original
    assert answer("deny")["approved"] is False
    assert len(cards) == 2
    for card in cards:
        assert card["pattern_key"] == approval._SECURITY_CONFIG_APPROVAL_KEY
        assert card["allow_session"] is False
        assert card["allow_permanent"] is False
    assert (home / "config.yaml").read_text() == original
