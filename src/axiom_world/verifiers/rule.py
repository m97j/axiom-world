"""Deterministic rule-tier verifiers (Tier 0/1 — sole reward authority).

Context contract: every PlayWorld verification receives
    context = {"scenario": <Scenario dict or object>}
and the prediction is the model's raw text, expected to contain a JSON object:
    {"actions": [{"type": "MOVE", "target": "loc_1"}, ...]}
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from axiom_world.core.enums import VerificationStatus
from axiom_world.verifiers.base import Verdict, Verifier
from axiom_world.worlds.playworld.engine import TransitionEngine
from axiom_world.worlds.playworld.spec import ACTION_GRAMMAR, Action, Scenario

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    match = _JSON_BLOCK.search(text)
    if match is None:
        return None
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _scenario_from_context(context: dict[str, Any]) -> Scenario:
    raw = context["scenario"]
    return raw if isinstance(raw, Scenario) else Scenario.model_validate(raw)


def parse_episode_actions(prediction: str) -> tuple[list[Action] | None, str]:
    """Shared parser. Returns (actions, reason_code)."""
    payload = _extract_json(prediction)
    if payload is None:
        return None, "malformed_json"
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        return None, "missing_actions_field"
    actions: list[Action] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            return None, "action_not_object"
        action_type = item.get("type")
        if action_type not in ACTION_GRAMMAR:
            return None, "unknown_action_type"
        try:
            actions.append(Action.model_validate(item))
        except ValidationError:
            return None, "action_schema_invalid"
    return actions, "ok"


class SchemaVerifier(Verifier):
    """Tier 0: output is parseable and grammar-conformant."""

    name = "schema"
    version = "1.0"

    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        actions, reason = parse_episode_actions(prediction)
        if actions is None:
            return self._make(VerificationStatus.FAILED, reason, score=0.0)
        return self._make(VerificationStatus.PASSED, "schema_valid", score=1.0,
                          action_count=len(actions))


class LegalityVerifier(Verifier):
    """Tier 1: every action legal under the transition engine (per-action rate)."""

    name = "legality"
    version = "1.0"

    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        actions, reason = parse_episode_actions(prediction)
        if actions is None:
            return self._make(VerificationStatus.SKIPPED, f"unparseable:{reason}")
        scenario = _scenario_from_context(context)
        engine = TransitionEngine(scenario.spec)
        state = scenario.initial_state
        legal_count = 0
        first_illegal: str | None = None
        for action in actions:
            result = engine.step(state, action)
            if not result.legal:
                first_illegal = first_illegal or result.reason_code
                break
            legal_count += 1
            assert result.next_state is not None
            state = result.next_state
        rate = legal_count / len(actions)
        if first_illegal is None:
            return self._make(VerificationStatus.PASSED, "all_actions_legal", score=1.0,
                              legal_action_rate=rate)
        return self._make(VerificationStatus.FAILED, first_illegal, score=rate,
                          legal_action_rate=rate, illegal_at=legal_count)


class GoalVerifier(Verifier):
    """Tier 1: full replay — goal-valid episode (the primary metric's core)."""

    name = "goal"
    version = "1.0"

    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        actions, reason = parse_episode_actions(prediction)
        if actions is None:
            return self._make(VerificationStatus.SKIPPED, f"unparseable:{reason}")
        scenario = _scenario_from_context(context)
        engine = TransitionEngine(scenario.spec)
        reached, reason_code, final_state, trace = engine.replay(scenario, actions)
        status = VerificationStatus.PASSED if reached else VerificationStatus.FAILED
        return self._make(status, reason_code, score=1.0 if reached else 0.0,
                          steps=len(trace), final_turn=final_state.turn)


class StateConsistencyVerifier(Verifier):
    """Tier 1: model's claimed final state (optional field) matches replay."""

    name = "state_consistency"
    version = "1.0"

    def _verify(self, prediction: str, context: dict[str, Any]) -> Verdict:
        payload = _extract_json(prediction)
        if payload is None:
            return self._make(VerificationStatus.SKIPPED, "unparseable:malformed_json")
        claimed = payload.get("final_state")
        if claimed is None:
            return self._make(VerificationStatus.SKIPPED, "no_claimed_state")
        actions, reason = parse_episode_actions(prediction)
        if actions is None:
            return self._make(VerificationStatus.SKIPPED, f"unparseable:{reason}")
        scenario = _scenario_from_context(context)
        engine = TransitionEngine(scenario.spec)
        _, _, final_state, _ = engine.replay(scenario, actions)
        mismatches = []
        if not isinstance(claimed, dict):
            return self._make(VerificationStatus.FAILED, "claimed_state_not_object", score=0.0)
        for key in ("location", "energy"):
            if key in claimed and claimed[key] != getattr(final_state, key):
                mismatches.append(key)
        if mismatches:
            return self._make(VerificationStatus.FAILED, "state_contradiction", score=0.0,
                              mismatched_fields=mismatches)
        return self._make(VerificationStatus.PASSED, "state_consistent", score=1.0)
