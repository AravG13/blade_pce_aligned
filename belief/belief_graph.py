"""
BDCA Belief Graph
=================
Core contribution: probabilistic world model.

Every fact has: confidence, timestamp, source.
Confidence decays over time.  Object location is a distribution,
not a point.  Teammate-proximity speeds up decay.

Contrast with CoELA: flat text memory entries (stale = wrong).
Contrast with REVECA: relevance-scored entries (stale = still wrong).
"""
from __future__ import annotations
import math
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import (
    BELIEF_DECAY_BASE, BELIEF_DECAY_TEAMMATE, BELIEF_MIN_CONFIDENCE,
    BELIEF_OBS_CONFIDENCE, BELIEF_MSG_CONFIDENCE,
)

# ── Fact types ────────────────────────────────────────────────────────────────
F_LOCATION    = "location"
F_HOLDING     = "holding"
F_CONT_STATE  = "container_state"
F_SUBGOAL     = "subgoal_status"
F_AGENT_ROOM  = "agent_room"
F_AGENT_TGT   = "agent_target"

SRC_OBS    = "observed"
SRC_MSG    = "message"
SRC_INF    = "inferred"
SRC_PRIOR  = "prior"
SRC_DECAY  = "decayed"


@dataclass
class BeliefFact:
    subject:      str
    predicate:    str
    value:        str
    confidence:   float
    last_updated: int
    source:       str
    # Full probability distribution over alternative values
    alternatives: Dict[str, float] = field(default_factory=dict)

    def decayed(self, current_step: int,
                teammate_nearby: bool = False) -> "BeliefFact":
        steps = current_step - self.last_updated
        if steps <= 0:
            return self
        rate = BELIEF_DECAY_TEAMMATE if teammate_nearby else BELIEF_DECAY_BASE
        new_conf = max(BELIEF_MIN_CONFIDENCE,
                       self.confidence * (rate ** steps))
        f = copy.copy(self)
        f.confidence   = new_conf
        f.source       = SRC_DECAY
        return f

    def to_dict(self) -> dict:
        return {
            "subject":      self.subject,
            "predicate":    self.predicate,
            "value":        self.value,
            "confidence":   round(self.confidence, 3),
            "last_updated": self.last_updated,
            "source":       self.source,
            "alternatives": {k: round(v, 3) for k, v in self.alternatives.items()},
        }


class BeliefGraph:
    """
    Full probabilistic world model for one agent.

    Key operations:
      update_from_observation(obs_dict, step)
      update_from_message(parsed_msg, step)
      apply_decay(current_step, teammate_last_room)
      get_confidence(subject, predicate, value=None)
      best_hypothesis(subject, predicate) -> (value, conf)
      summary() -> compact string for LLM prompt
    """

    def __init__(self, agent_id: str, goal_objects: List[str],
                 teammate_id: str = "teammate"):
        self.agent_id     = agent_id
        self.goal_objects = goal_objects
        self.teammate_id  = teammate_id
        self._facts: Dict[Tuple[str, str], BeliefFact] = {}
        self.history: List[dict] = []

        self.known_rooms:      List[str] = []
        self.known_containers: List[str] = []
        self.goal_location:    str = ""
        self.inspected:        set = set()
        self._name_to_inspected_id: set = set()  # container class names

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_priors(self, rooms: List[str],
                          containers: List[str],
                          goal_location: str) -> None:
        self.known_rooms      = rooms
        self.known_containers = containers
        self.goal_location    = goal_location

        n = max(len(rooms) + len(containers), 1)
        prior = 1.0 / n

        for obj in self.goal_objects:
            alts = {r: prior for r in rooms}
            alts.update({c: prior for c in containers})
            # normalise to 1
            total = sum(alts.values())
            alts  = {k: v / total for k, v in alts.items()}
            best  = max(alts, key=alts.get)
            self._put(BeliefFact(
                subject=obj, predicate="location",
                value=best, confidence=alts[best],
                last_updated=0, source=SRC_PRIOR,
                alternatives=alts,
            ))

        for c in containers:
            self._put(BeliefFact(
                subject=c, predicate="inspected",
                value="False", confidence=1.0,
                last_updated=0, source=SRC_PRIOR,
            ))

    # ── Internal setter ───────────────────────────────────────────────────────

    def _put(self, fact: BeliefFact) -> None:
        key = (fact.subject, fact.predicate)
        self._facts[key] = fact
        self.history.append(fact.to_dict())

    def _key(self, subject: str, predicate: str):
        return (subject, predicate)

    # ── Update: observation ───────────────────────────────────────────────────

    def update_from_observation(self, obs: dict, step: int) -> None:
        """
        obs expected keys:
          current_room, visible_objects, visible_containers,
          inspected_containers, held_objects,
          teammate_visible, teammate_id, teammate_room,
          teammate_holding, completed_subgoals
        """
        room = obs.get("current_room", "unknown")
        self._put(BeliefFact(self.agent_id, "location", room,
                             1.0, step, SRC_OBS))

        for obj in obs.get("visible_objects", []):
            # Use class_name as the belief subject so it matches goal_objects
            obj_class = obj.get("class_name", str(obj["id"]))
            self._update_location(obj_class, room, step, SRC_OBS,
                                  BELIEF_OBS_CONFIDENCE)
            # Also index by raw id for fine-grained tracking
            oid = str(obj["id"])
            if oid != obj_class:
                self._update_location(oid, room, step, SRC_OBS,
                                      BELIEF_OBS_CONFIDENCE)

        for cont in obs.get("visible_containers", []):
            cid  = cont["id"]
            state = cont.get("state", "CLOSED").upper()
            self._put(BeliefFact(cid, "container_state", state,
                                 BELIEF_OBS_CONFIDENCE, step, SRC_OBS))

        for cont in obs.get("inspected_containers", []):
            cid      = str(cont["id"])
            cname    = cont.get("class_name", "")
            contents = cont.get("contents", [])
            self.mark_inspected(cid, cname)
            self._put(BeliefFact(cid, "inspected", "True",
                                 1.0, step, SRC_OBS))
            for oid in contents:
                obj_class = oid if isinstance(oid, str) else str(oid)
                self._update_location(obj_class, cid, step, SRC_OBS, 0.98)
            if not contents:
                for obj in self.goal_objects:
                    fact = self._facts.get(self._key(obj, "location"))
                    if fact and fact.value == cid:
                        self._redistribute_away(obj, cid, step)

        for oid in obs.get("held_objects", []):
            self._put(BeliefFact(self.agent_id, f"holds_{oid}", "True",
                                 1.0, step, SRC_OBS))

        if obs.get("teammate_visible"):
            tm   = obs.get("teammate_id", self.teammate_id)
            troom = obs.get("teammate_room", room)
            self._put(BeliefFact(tm, "location", troom,
                                 BELIEF_OBS_CONFIDENCE, step, SRC_OBS))
            for oid in obs.get("teammate_holding", []):
                self._put(BeliefFact(tm, f"holds_{oid}", "True",
                                     BELIEF_OBS_CONFIDENCE, step, SRC_OBS))

        for sg in obs.get("completed_subgoals", []):
            self._put(BeliefFact(sg, "status", "complete",
                                 1.0, step, SRC_OBS))

    # ── Update: message ───────────────────────────────────────────────────────

    def update_from_message(self, msg: dict, step: int) -> None:
        """
        msg keys: type, object_id, room, sender, completed
        type in: subgoal_complete | object_found | intent | validation_reply
        """
        sender   = msg.get("sender", self.teammate_id)
        msg_type = msg.get("type", "")

        if msg_type == "subgoal_complete":
            oid = msg.get("object_id", "")
            if oid:
                self._put(BeliefFact(oid, "status", "complete",
                                     0.99, step, SRC_MSG))

        elif msg_type == "object_found":
            oid  = msg.get("object_id", "")
            room = msg.get("room", "")
            if oid and room:
                self._update_location(oid, room, step, SRC_MSG,
                                      BELIEF_MSG_CONFIDENCE)

        elif msg_type == "intent":
            room = msg.get("room", "")
            oid  = msg.get("object_id", "")
            if room:
                self._put(BeliefFact(sender, "target_room", room,
                                     0.80, step, SRC_MSG))
            if oid:
                self._put(BeliefFact(sender, "target_object", oid,
                                     0.80, step, SRC_MSG))

        elif msg_type == "validation_reply":
            oid       = msg.get("object_id", "")
            completed = msg.get("completed", False)
            if oid and completed:
                self._put(BeliefFact(oid, "status", "complete",
                                     0.97, step, SRC_MSG))

    # ── Temporal decay ────────────────────────────────────────────────────────

    def apply_decay(self, current_step: int,
                    teammate_last_room: Optional[str] = None) -> None:
        for key, fact in list(self._facts.items()):
            if fact.predicate not in ("location", "container_state"):
                continue
            if fact.last_updated == current_step:
                continue
            nearby = (teammate_last_room is not None
                      and fact.value == teammate_last_room)
            decayed = fact.decayed(current_step, nearby)
            if decayed.confidence != fact.confidence:
                self._facts[key] = decayed

    # ── Mutual-exclusivity location update ────────────────────────────────────

    def _update_location(self, obj_id: str, location: str,
                         step: int, source: str, conf: float) -> None:
        key      = self._key(obj_id, "location")
        existing = self._facts.get(key)
        old_alts = dict(existing.alternatives) if existing else {
            r: 1.0 / max(len(self.known_rooms), 1)
            for r in self.known_rooms
        }

        new_alts = dict(old_alts)
        new_alts[location] = conf
        other_keys   = [k for k in new_alts if k != location]
        old_other    = sum(old_alts.get(k, 0.0) for k in other_keys)
        remaining    = 1.0 - conf
        for k in other_keys:
            share       = old_alts.get(k, 0.0) / max(old_other, 1e-9)
            new_alts[k] = remaining * share

        self._put(BeliefFact(
            subject=obj_id, predicate="location",
            value=location, confidence=conf,
            last_updated=step, source=source,
            alternatives=new_alts,
        ))

    def _redistribute_away(self, obj_id: str,
                           excluded: str, step: int) -> None:
        key  = self._key(obj_id, "location")
        fact = self._facts.get(key)
        if not fact:
            return
        alts  = dict(fact.alternatives)
        freed = alts.pop(excluded, 0.0)
        if not alts:
            return
        total = sum(alts.values()) or 1e-9
        alts  = {k: v + freed * v / total for k, v in alts.items()}
        best  = max(alts, key=alts.get)
        new   = copy.copy(fact)
        new.value        = best
        new.confidence   = alts[best]
        new.alternatives = alts
        new.last_updated = step
        new.source       = SRC_INF
        self._put(new)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_confidence(self, subject: str, predicate: str,
                       value: Optional[str] = None) -> float:
        fact = self._facts.get(self._key(subject, predicate))
        if not fact:
            return 0.0
        if value is None or fact.value == value:
            return fact.confidence
        return fact.alternatives.get(value, 0.0)

    def best_hypothesis(self, subject: str,
                        predicate: str) -> Tuple[str, float]:
        fact = self._facts.get(self._key(subject, predicate))
        if not fact:
            return ("unknown", 0.0)
        return (fact.value, fact.confidence)

    def get_all_location_hypotheses(self, obj_id: str) -> Dict[str, float]:
        fact = self._facts.get(self._key(obj_id, "location"))
        if not fact:
            return {}
        d = dict(fact.alternatives)
        d[fact.value] = fact.confidence
        return d

    def get_subgoal_status(self, obj_id: str) -> str:
        fact = self._facts.get(self._key(obj_id, "status"))
        return "complete" if (fact and fact.value == "complete") else "incomplete"

    def is_inspected(self, container_id: str) -> bool:
        """Check by exact id OR by matching class name stored during observation."""
        if container_id in self.inspected:
            return True
        # Also check if this name maps to an inspected numeric id
        return container_id in self._name_to_inspected_id

    def mark_inspected(self, container_id: str, container_name: str = "") -> None:
        """Mark a container as inspected by both its id and name."""
        self.inspected.add(str(container_id))
        if container_name:
            self._name_to_inspected_id.add(container_name)

    def uncertain_objects(self, threshold: float = 0.70) -> List[str]:
        """Goal objects whose best-hypothesis confidence < threshold."""
        return [
            obj for obj in self.goal_objects
            if self.get_confidence(obj, "location") < threshold
        ]

    def total_entropy(self) -> float:
        total = 0.0
        for obj in self.goal_objects:
            alts = self.get_all_location_hypotheses(obj)
            if not alts:
                continue
            s     = sum(alts.values()) or 1e-9
            probs = [v / s for v in alts.values()]
            total += -sum(p * math.log2(p + 1e-12) for p in probs)
        return total

    def summary(self) -> str:
        """Compact structured string fed to the LLM planner."""
        lines = ["[BeliefGraph]"]
        for obj in self.goal_objects:
            loc, conf = self.best_hypothesis(obj, "location")
            status    = self.get_subgoal_status(obj)
            lines.append(
                f"  {obj}: loc={loc}({conf:.0%}) status={status}"
            )
        tm_loc, tm_conf = self.best_hypothesis(self.teammate_id, "location")
        tm_tgt, tgt_c   = self.best_hypothesis(self.teammate_id, "target_object")
        lines.append(
            f"  teammate: room={tm_loc}({tm_conf:.0%}) "
            f"likely_target={tm_tgt}({tgt_c:.0%})"
        )
        uninspected = [
            c for c in self.known_containers if not self.is_inspected(c)
        ]
        lines.append(f"  uninspected_containers: {uninspected}")
        lines.append(f"  entropy: {self.total_entropy():.2f}")
        return "\n".join(lines)
