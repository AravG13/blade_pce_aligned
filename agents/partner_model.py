"""
BDCA Partner Intent Model
=========================
Proactively predicts teammate room, held objects, and target subgoal
at every planning step.  Output feeds directly into action scoring
as a collision-risk penalty — preventing bad plans before selection,
not catching them after (unlike REVECA's trajectory validation).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import PARTNER_CACHE_STEPS, COLLISION_RISK_THRESHOLD


@dataclass
class PartnerPrediction:
    likely_room:        str
    room_conf:          float
    likely_held:        List[str]
    likely_target:      str
    target_conf:        float
    collision_risk:     Dict[str, float] = field(default_factory=dict)
    steps_since_seen:   int = 0
    predicted_at:       int = 0

    def is_fresh(self, step: int) -> bool:
        return (step - self.predicted_at) < PARTNER_CACHE_STEPS

    def to_dict(self) -> dict:
        return {
            "likely_room":      self.likely_room,
            "room_conf":        round(self.room_conf, 3),
            "likely_held":      self.likely_held,
            "likely_target":    self.likely_target,
            "target_conf":      round(self.target_conf, 3),
            "collision_risk":   {k: round(v, 3)
                                 for k, v in self.collision_risk.items()},
        }

    def __repr__(self):
        return (f"PartnerPrediction(room={self.likely_room}[{self.room_conf:.0%}],"
                f" target={self.likely_target}[{self.target_conf:.0%}])")


class PartnerIntentModel:
    """
    Hybrid rule-based + LLM partner predictor.

    Rule layer: fast, deterministic heuristics from belief graph.
    LLM layer:  richer reasoning when state is genuinely ambiguous.
                Skipped when cached result is fresh.
    """

    def __init__(self, agent_id: str, teammate_id: str,
                 goal_objects: List[str], llm_fn=None):
        self.agent_id     = agent_id
        self.teammate_id  = teammate_id
        self.goal_objects = goal_objects
        self.llm_fn       = llm_fn   # callable(prompt_str) -> str

        self._cache:    Optional[PartnerPrediction] = None
        self._log:      List[dict] = []
        # Forced collision risks from received commitment broadcasts:
        # {object_class: expiry_step} — set to 1.0 until expiry
        self._forced_risk: Dict[str, int] = {}

    # ── Forced collision risk (commitment broadcasts) ─────────────────────────

    def force_collision_risk(self, obj_class: str, expiry_step: int) -> None:
        """Pin collision_risk for obj_class to 1.0 until expiry_step."""
        self._forced_risk[obj_class] = expiry_step
        # Invalidate cache so forced override is applied immediately
        self._cache = None

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(self, belief_graph,
                conversation_log: List[dict],
                current_step: int,
                incomplete_goals: List[str]) -> PartnerPrediction:

        if self._cache and self._cache.is_fresh(current_step):
            return self._cache

        ctx = self._build_context(belief_graph, conversation_log,
                                  current_step, incomplete_goals)

        # Only call LLM when genuinely uncertain AND not too frequent
        steps_since_llm = current_step - getattr(self, "_last_llm_step", -99)
        if (self.llm_fn and ctx["uncertainty_high"]
                and steps_since_llm >= 8):
            pred = self._llm_predict(ctx, current_step)
            self._last_llm_step = current_step
        else:
            pred = self._rule_predict(ctx, current_step)

        pred.collision_risk = self._collision_risk(
            pred, belief_graph, incomplete_goals)

        # Apply commitment-broadcast overrides: partner explicitly targeting
        # this class → collision risk is certain until the commitment expires
        expired = [cls for cls, exp in self._forced_risk.items()
                   if current_step >= exp]
        for cls in expired:
            del self._forced_risk[cls]
        for cls, exp in self._forced_risk.items():
            pred.collision_risk[cls] = 1.0

        self._cache = pred
        self._log.append({"step": current_step, "pred": pred.to_dict()})
        return pred

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, bg, conversation_log: List[dict],
                       current_step: int,
                       incomplete_goals: List[str]) -> dict:
        tm_room, tm_room_conf  = bg.best_hypothesis(self.teammate_id, "location")
        tm_tgt,  tm_tgt_conf   = bg.best_hypothesis(self.teammate_id, "target_object")

        # Last seen step
        fact = bg.get_confidence(self.teammate_id, "location")
        steps_since = current_step  # conservative if never seen

        held = [
            obj for obj in self.goal_objects
            if bg.get_confidence(self.teammate_id, f"holds_{obj}") > 0.4
        ]

        # Object-proximity heuristic: which incomplete goal object is closest
        # to the teammate's believed room?
        proximate_obj = None
        best_conf     = 0.0
        for obj in incomplete_goals:
            obj_loc, obj_conf = bg.best_hypothesis(obj, "location")
            if obj_loc == tm_room and obj_conf > best_conf:
                proximate_obj = obj
                best_conf     = obj_conf

        recent_msgs = conversation_log[-3:] if conversation_log else []

        return {
            "tm_room":          tm_room,
            "tm_room_conf":     tm_room_conf,
            "tm_tgt":           tm_tgt,
            "tm_tgt_conf":      tm_tgt_conf,
            "tm_held":          held,
            "steps_since":      steps_since,
            "proximate_obj":    proximate_obj,
            "proximate_conf":   best_conf,
            "incomplete_goals": incomplete_goals,
            "recent_msgs":      recent_msgs,
            "uncertainty_high": (tm_room_conf < 0.60
                                 or tm_tgt_conf < 0.50),
        }

    # ── Rule-based prediction (fast, no LLM) ─────────────────────────────────

    def _rule_predict(self, ctx: dict, step: int) -> PartnerPrediction:
        """
        Deterministic heuristic rules:
        1. If teammate last messaged about a room, they are likely there.
        2. If a high-confidence goal object is in their last known room,
           they are likely targeting it.
        3. Otherwise, fall back to last known state.
        """
        room       = ctx["tm_room"]
        room_conf  = ctx["tm_room_conf"]
        target     = ctx["proximate_obj"] or ctx["tm_tgt"] or "unknown"
        target_conf = ctx["proximate_conf"] if ctx["proximate_obj"] else ctx["tm_tgt_conf"]

        # Boost: if recent message mentions a room, update room belief
        for msg in reversed(ctx["recent_msgs"]):
            content = msg.get("content", "")
            for r in ["kitchen", "livingroom", "bedroom", "bathroom"]:
                if r in content.lower():
                    room      = r
                    room_conf = max(room_conf, 0.75)
                    break

        return PartnerPrediction(
            likely_room   = room,
            room_conf     = room_conf,
            likely_held   = ctx["tm_held"],
            likely_target = target,
            target_conf   = target_conf,
            steps_since_seen = ctx["steps_since"],
            predicted_at  = step,
        )

    # ── LLM prediction (richer, slower) ──────────────────────────────────────

    def _llm_predict(self, ctx: dict, step: int) -> PartnerPrediction:
        prompt = f"""You are predicting the current state of a cooperative agent's teammate.

Evidence:
- Teammate last known room: {ctx['tm_room']} (confidence {ctx['tm_room_conf']:.0%})
- Teammate last known target: {ctx['tm_tgt']} (confidence {ctx['tm_tgt_conf']:.0%})
- Teammate held objects: {ctx['tm_held']}
- Steps since last seen: {ctx['steps_since']}
- Incomplete subgoals: {ctx['incomplete_goals']}
- Recent messages: {ctx['recent_msgs']}

Predict the teammate's CURRENT state. Output ONLY valid JSON with these exact keys:
{{
  "likely_room": "<room name>",
  "room_confidence": <0.0-1.0>,
  "likely_target": "<object name or unknown>",
  "target_confidence": <0.0-1.0>
}}"""

        try:
            raw    = self.llm_fn(prompt)
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            return PartnerPrediction(
                likely_room   = data.get("likely_room", ctx["tm_room"]),
                room_conf     = float(data.get("room_confidence",
                                               ctx["tm_room_conf"])),
                likely_held   = ctx["tm_held"],
                likely_target = data.get("likely_target", ctx["tm_tgt"]),
                target_conf   = float(data.get("target_confidence",
                                               ctx["tm_tgt_conf"])),
                steps_since_seen = ctx["steps_since"],
                predicted_at  = step,
            )
        except Exception:
            return self._rule_predict(ctx, step)

    # ── Collision risk ────────────────────────────────────────────────────────

    def _collision_risk(self, pred: PartnerPrediction,
                        belief_graph,
                        incomplete_goals: List[str]) -> Dict[str, float]:
        """
        For each incomplete goal object compute probability that
        teammate is already handling it.

        risk(obj) = P(teammate in same room as obj)
                    * P(obj is high-relevance to teammate)
                    * decay_factor
        """
        risk = {}
        for obj in incomplete_goals:
            obj_loc, obj_conf = belief_graph.best_hypothesis(obj, "location")
            same_room  = 1.0 if obj_loc == pred.likely_room else 0.0
            # Decay risk if confidence in teammate room is low
            room_factor = pred.room_conf
            is_target   = 1.0 if obj == pred.likely_target else 0.3
            r = same_room * room_factor * is_target * obj_conf
            risk[obj] = min(r, 1.0)
        return risk
