"""
BDCA Belief-Aware Planner
=========================
Generates candidate actions from the belief graph,
scores each one using:
  - Expected goal progress (belief-weighted)
  - Information gain
  - Travel cost
  - Redundancy penalty (partner collision risk)

Then feeds top-K scored candidates to the LLM for final selection.
LLM's job: reason among pre-filtered, pre-scored options.
This is fundamentally different from CoELA (LLM sees everything)
and REVECA (LLM sees top-K by relevance only, no uncertainty weighting).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import (
    WEIGHT_GOAL_PROGRESS, WEIGHT_INFO_GAIN,
    WEIGHT_TRAVEL_COST, WEIGHT_REDUNDANCY,
    ACT_EXPLORE, ACT_CHECK, ACT_GRAB, ACT_PUT, ACT_MSG,
    TOP_K_CANDIDATES,
)


@dataclass
class CandidateAction:
    action_str:     str     # "[gograb] <apple> (375)"
    action_type:    str     # ACT_GRAB etc.
    target_id:      str
    target_class:   str
    score:          float
    goal_progress:  float
    info_gain:      float
    travel_cost:    float
    redundancy:     float

    def to_prompt_line(self, letter: str) -> str:
        return (
            f"{letter}. {self.action_str}  "
            f"[score={self.score:.2f}, "
            f"goal={self.goal_progress:.2f}, "
            f"info_gain={self.info_gain:.2f}, "
            f"redundancy_risk={self.redundancy:.2f}]"
        )


class BeliefAwarePlanner:
    """
    Generates + scores candidate actions.
    Returns a ranked list; top-K go to the LLM.
    """

    def __init__(self, agent_id: str, goal_objects: List[str],
                 rooms: List[str], llm_fn=None,
                 goal_location_class: str = ""):
        self.agent_id            = agent_id
        self.goal_objects        = goal_objects
        self.rooms               = rooms
        self.llm_fn              = llm_fn
        self.goal_location_class = goal_location_class
        self._plan_log: List[dict] = []

    # ── Main planning call ────────────────────────────────────────────────────

    def plan(self,
             belief_graph,
             partner_pred,
             obs_parsed: dict,
             incomplete_goals: List[str],
             action_history: List[str],
             current_step: int,
             comm_decision=None) -> Tuple[str, dict]:
        """
        Returns (selected_action_str, info_dict).
        """
        candidates = self._generate_candidates(
            belief_graph, partner_pred, obs_parsed, incomplete_goals)

        if not candidates:
            return "[wait]", {"reason": "no_candidates"}

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        top_k = candidates[:TOP_K_CANDIDATES]

        # Add communication action if VOI policy suggests it
        if comm_decision and comm_decision.should_send:
            comm_cand = CandidateAction(
                action_str   = f"[send_message] <{comm_decision.content}>",
                action_type  = ACT_MSG,
                target_id    = "",
                target_class = "message",
                score        = comm_decision.voi,
                goal_progress=0.0,
                info_gain    = comm_decision.voi + 1.0,
                travel_cost  = 1.0,
                redundancy   = 0.0,
            )
            top_k.append(comm_cand)

        if self.llm_fn:
            selected, llm_info = self._llm_select(
                top_k, belief_graph, obs_parsed,
                incomplete_goals, action_history, current_step)
        else:
            selected  = top_k[0].action_str
            llm_info  = {"method": "greedy_score"}

        info = {
            "candidates":     [c.action_str for c in top_k],
            "scores":         [round(c.score, 3) for c in top_k],
            "selected":       selected,
            "step":           current_step,
        }
        info.update(llm_info)
        self._plan_log.append(info)
        return selected, info

    # ── Candidate generation ──────────────────────────────────────────────────

    def _generate_candidates(self,
                             belief_graph,
                             partner_pred,
                             obs_parsed: dict,
                             incomplete_goals: List[str]
                             ) -> List[CandidateAction]:
        candidates = []
        my_room    = obs_parsed.get("current_room", "unknown")
        held       = obs_parsed.get("held_objects", [])
        n_held     = len(held)
        # Object IDs committed to by either agent — skip as grab candidates
        committed  = set(obs_parsed.get("committed_targets", {}).keys())

        # ── PUT action — if holding anything, goput is always an option ──
        if n_held > 0:
            gl = obs_parsed.get("goal_location", "coffeetable (999)")
            cand = CandidateAction(
                action_str   = f"[goput] {gl}",
                action_type  = ACT_PUT,
                target_id    = gl.split("(")[-1].rstrip(")"),
                target_class = gl.split()[0],
                score        = 0.0,
                goal_progress= 0.85,
                info_gain    = 0.0,
                travel_cost  = self._travel_cost(my_room, "goal"),
                redundancy   = 0.0,
            )
            cand.score = self._score(cand)
            candidates.append(cand)

        # ── GRAB actions — visible goal objects with low redundancy ──────
        if n_held < 2:
            for obj_info in obs_parsed.get("visible_objects", []):
                oid        = str(obj_info.get("id", ""))
                oclass     = obj_info.get("class_name", "")
                if oclass not in self.goal_objects:
                    continue
                if oclass not in incomplete_goals:
                    continue
                if oid in committed:
                    continue  # partner broadcast targeting this object
                redundancy = partner_pred.collision_risk.get(oclass, 0.0)
                loc_conf   = belief_graph.get_confidence(oclass, "location",
                                                          my_room)
                cand = CandidateAction(
                    action_str   = f"[gograb] <{oclass}> ({oid})",
                    action_type  = ACT_GRAB,
                    target_id    = oid,
                    target_class = oclass,
                    score        = 0.0,
                    goal_progress= loc_conf * 0.9,
                    info_gain    = 0.0,
                    travel_cost  = 2.0,  # already in room
                    redundancy   = redundancy,
                )
                cand.score = self._score(cand)
                candidates.append(cand)

            # Also add grab candidates from ungrabbed_objects tracker
            # (objects the agent knows about from previous room visits)
            for obj_list in obs_parsed.get("ungrabbed_by_room", {}).values():
                for obj_info in (obj_list or []):
                    oid    = str(obj_info.get("id", ""))
                    oclass = obj_info.get("class_name", "")
                    if oclass not in incomplete_goals:
                        continue
                    if oid in committed:
                        continue  # partner broadcast targeting this object
                    if any(c.target_id == oid for c in candidates):
                        continue  # already added as visible
                    redundancy = partner_pred.collision_risk.get(oclass, 0.0)
                    loc, conf  = belief_graph.best_hypothesis(oclass, "location")
                    tcost      = self._travel_cost(my_room, loc)
                    cand = CandidateAction(
                        action_str   = f"[gograb] <{oclass}> ({oid})",
                        action_type  = ACT_GRAB,
                        target_id    = oid,
                        target_class = oclass,
                        score        = 0.0,
                        goal_progress= conf * 0.85,
                        info_gain    = 0.0,
                        travel_cost  = tcost,
                        redundancy   = redundancy,
                    )
                    cand.score = self._score(cand)
                    candidates.append(cand)

        # ── CHECK actions — uninspected containers (all known, not just visible) ──
        if n_held < 2:
            # First: visible containers in current room
            for cont_info in obs_parsed.get("visible_containers", []):
                cid    = str(cont_info.get("id", ""))
                cclass = cont_info.get("class_name", "")
                if belief_graph.is_inspected(cid) or belief_graph.is_inspected(cclass):
                    continue
                exp_objects = sum(
                    belief_graph.get_confidence(obj, "location", cid) +
                    belief_graph.get_confidence(obj, "location", cclass)
                    for obj in incomplete_goals
                )
                if exp_objects < 0.05 and cclass != self.goal_location_class:
                    continue
                info_gain = exp_objects * 0.8
                cand = CandidateAction(
                    action_str   = f"[gocheck] <{cclass}> ({cid})",
                    action_type  = ACT_CHECK,
                    target_id    = cid,
                    target_class = cclass,
                    score        = 0.0,
                    goal_progress= exp_objects * 0.6,
                    info_gain    = info_gain,
                    travel_cost  = 3.0,
                    redundancy   = 0.0,
                )
                cand.score = self._score(cand)
                candidates.append(cand)

            # Second: known containers in OTHER rooms (cross-room check candidates)
            all_rooms_explored = set(obs_parsed.get("explored_rooms", [])) >= set(self.rooms)
            search_partition   = set(obs_parsed.get("search_partition", self.rooms))
            for cont_info in obs_parsed.get("all_unchecked_containers", []):
                cid    = str(cont_info.get("id", ""))
                cclass = cont_info.get("class_name", "")
                if belief_graph.is_inspected(cid) or belief_graph.is_inspected(cclass):
                    continue
                if any(c.target_id == cid for c in candidates):
                    continue  # already added as visible
                exp_objects = sum(
                    belief_graph.get_confidence(obj, "location", cid) +
                    belief_graph.get_confidence(obj, "location", cclass)
                    for obj in incomplete_goals
                )
                # When all rooms explored, check ALL remaining containers regardless of score
                if exp_objects < 0.08 and not all_rooms_explored:
                    continue
                cont_room  = cont_info.get("room", my_room)
                # Skip containers in partner's partition unless own partition exhausted
                own_partition_done = search_partition and cont_room not in search_partition
                if own_partition_done and not all_rooms_explored:
                    continue
                # Boost info_gain when all rooms exhausted — must check containers
                urgency    = 1.5 if all_rooms_explored else 1.0
                info_gain  = max(exp_objects * 0.9, 0.3) * urgency
                goal_prog  = max(exp_objects * 0.7, 0.2) * urgency
                cand = CandidateAction(
                    action_str   = f"[gocheck] <{cclass}> ({cid})",
                    action_type  = ACT_CHECK,
                    target_id    = cid,
                    target_class = cclass,
                    score        = 0.0,
                    goal_progress= goal_prog,
                    info_gain    = info_gain,
                    travel_cost  = self._travel_cost(my_room, cont_room),
                    redundancy   = 0.0,
                )
                cand.score = self._score(cand)
                candidates.append(cand)

        # ── EXPLORE actions — rooms with uncertain goal objects ────────────
        explored_rooms   = set(obs_parsed.get("explored_rooms", []))
        exhausted_rooms  = set(obs_parsed.get("exhausted_rooms", []))
        search_partition = obs_parsed.get("search_partition", self.rooms)
        for room in self.rooms:
            if room == my_room:
                continue
            if room in exhausted_rooms:
                continue
            # Prefer own search partition — deprioritise partner's partition
            in_partition = room in search_partition
            room_value = sum(
                belief_graph.get_confidence(obj, "location", room)
                for obj in incomplete_goals
            )
            if room_value < 0.05:
                continue

            never_visited  = room not in explored_rooms
            in_partition   = room in search_partition
            info_gain      = room_value * 0.4 + (1.2 if never_visited else 0.0)
            goal_progress  = room_value * 0.5 + (0.5 if never_visited else 0.0)
            # Penalise rooms outside own partition — let partner handle those
            partition_penalty = 0.0 if in_partition else 0.4
            redundancy     = max(partition_penalty,
                                 0.5 if room == partner_pred.likely_room else 0.0)
            room_id        = obs_parsed.get("room_ids", {}).get(room, 0)

            cand = CandidateAction(
                action_str   = f"[goexplore] <{room}> ({room_id})",
                action_type  = ACT_EXPLORE,
                target_id    = str(room_id),
                target_class = room,
                score        = 0.0,
                goal_progress= goal_progress,
                info_gain    = info_gain,
                travel_cost  = self._travel_cost(my_room, room),
                redundancy   = redundancy,
            )
            cand.score = self._score(cand)
            candidates.append(cand)

        return candidates

    def _score(self, c: CandidateAction) -> float:
        return (
            WEIGHT_GOAL_PROGRESS * c.goal_progress
            + WEIGHT_INFO_GAIN   * c.info_gain
            - WEIGHT_TRAVEL_COST * c.travel_cost
            - WEIGHT_REDUNDANCY  * c.redundancy
        )

    def _travel_cost(self, from_room: str, to_room: str) -> float:
        if from_room == to_room or to_room == "goal":
            return 5.0
        return 15.0

    # ── LLM selection ─────────────────────────────────────────────────────────

    def _llm_select(self,
                    candidates: List[CandidateAction],
                    belief_graph,
                    obs_parsed: dict,
                    incomplete_goals: List[str],
                    action_history: List[str],
                    step: int) -> Tuple[str, dict]:

        belief_summary = belief_graph.summary()
        history_str    = ", ".join(action_history[-6:])
        held_str       = ", ".join(obs_parsed.get("held_objects", [])) or "nothing"
        room           = obs_parsed.get("current_room", "unknown")

        options = "\n".join(
            c.to_prompt_line(chr(ord("A") + i))
            for i, c in enumerate(candidates)
        )

        prompt = f"""You are a cooperative embodied agent. Choose the best next action.

Agent: {self.agent_id}
Current room: {room}
Holding: {held_str}
Step: {step}/250
Remaining subgoals: {incomplete_goals}
Recent actions: {history_str}

{belief_summary}

Candidate actions (pre-scored by belief-weighted utility):
{options}

Rules:
- Prefer high goal_progress over exploration when objects are known.
- Avoid high redundancy_risk actions (teammate may already handle).
- If holding objects and goal location is reachable, PUT is usually best.

Answer with the SINGLE letter of the best action. Then one sentence of reasoning.
Format: "X. <reasoning>"
"""

        try:
            raw = self.llm_fn(prompt).strip()
            letter = raw[0].upper()
            idx    = ord(letter) - ord("A")
            if 0 <= idx < len(candidates):
                return candidates[idx].action_str, {
                    "llm_output": raw,
                    "selected_idx": idx,
                }
        except Exception as e:
            pass

        # Fallback: return highest-scored
        return candidates[0].action_str, {"method": "fallback_greedy"}
