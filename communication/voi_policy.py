"""
BDCA VOI Communication Policy
==============================
Decides WHETHER and WHAT to communicate.
Key: a message is sent only when VOI > threshold.

VOI(msg) = expected_waste_avoided - step_cost

CoELA: always generates a message, then decides whether to send.
REVECA: sends in 4 fixed cases (init / validation / reply / complete).
BDCA: sends when information value exceeds cost.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

from config import (
    COMM_STEP_COST, VOI_THRESHOLD, COMM_COOLDOWN, MAX_MSG_LENGTH,
)


@dataclass
class MessageDecision:
    should_send:  bool
    message_type: str     # query | broadcast | subgoal_complete | none
    content:      str
    voi:          float
    target_obj:   str = ""
    target_room:  str = ""


class VOICommunicationPolicy:
    """
    Expected-value communication policy.

    At each step, evaluates:
      1. Query messages — "have you handled X?"
      2. Broadcast messages — "I found X in room R"
      3. Subgoal-complete announcements (always sent, zero-cost signal)

    Returns the highest-VOI message, or a no-send decision.
    """

    def __init__(self, agent_id: str, teammate_id: str,
                 goal_objects: List[str], llm_fn=None):
        self.agent_id    = agent_id
        self.teammate_id = teammate_id
        self.goal_objects = goal_objects
        self.llm_fn      = llm_fn

        self._last_sent_step = -999
        self._sent_log: List[dict] = []

    # ── Main decision ─────────────────────────────────────────────────────────

    def decide(self, belief_graph,
               partner_pred,
               current_step: int,
               incomplete_goals: List[str],
               just_completed: Optional[str] = None,
               my_room: str = "",
               just_targeted: Optional[str] = None,
               just_exhausted: Optional[str] = None) -> MessageDecision:

        # Commitment broadcast — agent just selected a [gograb] target.
        # Bypass cooldown entirely: partner needs this immediately to avoid
        # wasting steps walking to the same object.
        # just_targeted format: "object_class:object_id"
        if just_targeted:
            return MessageDecision(
                should_send  = True,
                message_type = "commitment",
                content      = f"targeting:{just_targeted}",
                voi          = 999.0,
                target_obj   = just_targeted.split(":")[0],
            )

        # Room-exhaustion broadcast — partner can stop planning to search this room.
        # Saves 8-15 real-simulator steps per avoided redundant room visit.
        # just_exhausted is the plain room name, e.g. "kitchen"
        if just_exhausted:
            return MessageDecision(
                should_send  = True,
                message_type = "room_exhausted",
                content      = f"exhausted:{just_exhausted}",
                voi          = 999.0,
                target_room  = just_exhausted,
            )

        # Always announce subgoal completions (they're free in information terms)
        if just_completed:
            return MessageDecision(
                should_send  = True,
                message_type = "subgoal_complete",
                content      = (f"I completed subgoal: {just_completed}. "
                                f"I'm in {my_room}."),
                voi          = 999.0,
                target_obj   = just_completed,
            )

        # Cooldown guard
        if current_step - self._last_sent_step < COMM_COOLDOWN:
            return MessageDecision(False, "none", "", 0.0)

        best = MessageDecision(False, "none", "", 0.0)

        # ── Evaluate query messages ────────────────────────────────────────
        for obj in incomplete_goals:
            risk = partner_pred.collision_risk.get(obj, 0.0)
            if risk < 0.3:
                continue  # Teammate almost certainly not handling it

            # Expected duplicate-work cost if both go for same object
            my_dist  = self._estimate_distance(belief_graph, obj, my_room)
            tm_dist  = self._estimate_distance(
                belief_graph, obj, partner_pred.likely_room)
            avg_wasted_steps = (my_dist + tm_dist) / 2.0

            # Uncertainty reduction from asking
            uncertainty = risk * (1 - risk)
            voi = (uncertainty * avg_wasted_steps) - COMM_STEP_COST

            if voi > best.voi:
                msg = self._format_query(obj, belief_graph, my_room)
                best = MessageDecision(
                    should_send  = voi > VOI_THRESHOLD,
                    message_type = "query",
                    content      = msg,
                    voi          = voi,
                    target_obj   = obj,
                )

        # ── Evaluate broadcast messages ────────────────────────────────────
        for obj in incomplete_goals:
            loc, conf = belief_graph.best_hypothesis(obj, "location")
            if conf < 0.75:
                continue  # Not confident enough to broadcast
            if loc == my_room:
                # I can see it — is teammate closer?
                tm_dist = self._estimate_distance(
                    belief_graph, obj, partner_pred.likely_room)
                my_dist = self._estimate_distance(
                    belief_graph, obj, my_room)
                if tm_dist < my_dist and partner_pred.room_conf > 0.5:
                    tm_gain = max(0, my_dist - tm_dist)
                    voi     = tm_gain - COMM_STEP_COST
                    if voi > best.voi:
                        msg = self._format_broadcast(obj, loc, my_room)
                        best = MessageDecision(
                            should_send  = voi > VOI_THRESHOLD,
                            message_type = "broadcast",
                            content      = msg,
                            voi          = voi,
                            target_obj   = obj,
                            target_room  = loc,
                        )

        if best.should_send:
            self._last_sent_step = current_step
            self._sent_log.append({
                "step":    current_step,
                "type":    best.message_type,
                "voi":     round(best.voi, 3),
                "target":  best.target_obj,
            })

        return best

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _estimate_distance(self, belief_graph,
                           obj: str, from_room: str) -> float:
        """
        Simple topological distance estimate.
        Returns expected steps to reach object from from_room.
        In C-WAH rooms are adjacent, so cross-room = ~15 steps,
        same room = ~5 steps.
        """
        obj_loc, conf = belief_graph.best_hypothesis(obj, "location")
        if obj_loc == "unknown":
            return 20.0  # high cost for unknown
        if obj_loc == from_room:
            return 5.0
        return 15.0  # cross-room travel

    def _format_query(self, obj: str, belief_graph,
                      my_room: str) -> str:
        obj_loc, conf = belief_graph.best_hypothesis(obj, "location")
        return (f"Hi, I'm in {my_room}. Have you already handled "
                f"the {obj} (last seen in {obj_loc})? "
                f"Please reply yes/no.")[:MAX_MSG_LENGTH]

    def _format_broadcast(self, obj: str, location: str,
                          my_room: str) -> str:
        return (f"I found {obj} in {location}. "
                f"I'm currently in {my_room} — "
                f"you may be closer, feel free to grab it.")[:MAX_MSG_LENGTH]

    def generate_natural_language(self, decision: MessageDecision) -> str:
        """
        If an LLM is available, convert structured intent to natural language.
        Falls back to the pre-formatted content string.
        """
        if not self.llm_fn or not decision.should_send:
            return decision.content

        prompt = (
            f"Convert this structured cooperative agent message to natural, "
            f"brief English (max 100 words):\n{decision.content}"
        )
        try:
            return self.llm_fn(prompt).strip()[:MAX_MSG_LENGTH]
        except Exception:
            return decision.content
