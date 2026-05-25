"""
BDCA Agent
==========
Wires all modules together into the CoELA-compatible agent interface.

reset(obs, containers_name, goal_objects_name, rooms_name, room_info, goal)
get_action(observation, goal) -> (action_str, info_dict)

Internally runs the full BDCA loop each step:
  1. Parse observation -> structured obs_dict
  2. Update belief graph
  3. Apply temporal decay
  4. Predict partner state
  5. Generate + score candidate actions
  6. Evaluate VOI for communication
  7. LLM selects among top-K
  8. Execute via CoELA-compatible skill dispatcher
"""
from __future__ import annotations
import os, json, requests
from typing import Dict, List, Optional, Tuple

from belief.belief_graph import BeliefGraph
from agents.partner_model import PartnerIntentModel
from planner.belief_planner import BeliefAwarePlanner
from communication.voi_policy import VOICommunicationPolicy
COMMITMENT_DURATION = 10  # steps a gograb commitment blocks partners

from config import (
    MAX_STEPS, MAX_HELD, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    CWAH_ROOMS, BELIEF_DECAY_BASE, BELIEF_DECAY_TEAMMATE,
    BELIEF_OBS_CONFIDENCE, BELIEF_MSG_CONFIDENCE,
    WEIGHT_GOAL_PROGRESS, WEIGHT_INFO_GAIN,
    WEIGHT_TRAVEL_COST, WEIGHT_REDUNDANCY,
    PARTNER_CACHE_STEPS, VOI_THRESHOLD,
    MAX_MSG_LENGTH, TOP_K_CANDIDATES,
    USE_BELIEF_GRAPH, USE_PARTNER_MODEL,
    USE_VOI_COMM, USE_BELIEF_DECAY,
)


class BDCAAgent:
    """
    Drop-in replacement for CoELA's LLM_agent.
    Same external interface, completely different internals.
    """

    def __init__(self, agent_id: int, char_index: int, args=None):
        self.agent_id           = agent_id
        self.agent_name         = "Alice" if agent_id == 1 else "Bob"
        self.opponent_agent_id  = 3 - agent_id
        self.opponent_name      = "Bob" if agent_id == 1 else "Alice"

        # ── State ────────────────────────────────────────────────────────────
        self.steps              = 0
        self.plan               = None      # current high-level plan string
        self.stuck              = 0
        self.current_room       = None      # dict node
        self.last_room          = None
        self.last_action        = None

        self.grabbed_objects:   List[int]  = []
        self.opponent_grabbed:  List[dict] = []
        self.reachable_objects: List[str]  = []
        self.id2node:           Dict       = {}
        self.id_inside_room:    Dict       = {}
        self.satisfied:         List[dict] = []
        self.unsatisfied:       dict       = {}
        self.rooms_name:        List[str]  = []
        self.roomname2id:       Dict       = {}
        self.containers_name:   List[str]  = []
        self.goal_objects_name: List[str]  = []
        self.goal_location:     str        = ""
        self.goal_location_id:  int        = -1
        self.unchecked_containers: Dict    = {}
        self.ungrabbed_objects:    Dict    = {}
        self.dialogue_history:  List[str]  = []
        self.action_history:    List[str]  = []
        self._prev_satisfied:      List[dict] = []
        self._broadcast_exhausted: set        = set()
        self._broadcast_grab:      set        = set()

        # ── LLM client ───────────────────────────────────────────────────────
        self._api_key = (getattr(args, "openai_api_key", None)
                         or os.environ.get("OPENAI_API_KEY", ""))
        self._llm_calls = 0

        # ── BDCA modules — initialised in reset() ────────────────────────────
        self.belief:        Optional[BeliefGraph]           = None
        self.partner_model: Optional[PartnerIntentModel]    = None
        self.planner:       Optional[BeliefAwarePlanner]    = None
        self.comm_policy:   Optional[VOICommunicationPolicy] = None

        # ── Container open state ─────────────────────────────────────────────
        # Container node IDs we have successfully sent [open] to this episode.
        # Persists across observation updates so stale obs never re-triggers [open].
        self.opened_containers: set = set()

        # ── Commitment tracking ───────────────────────────────────────────────
        # Maps str(object_id) -> expiry_step for objects committed to by either
        # agent via "targeting:" broadcasts.  Planner excludes these as grabs.
        self.committed_targets: Dict[str, int] = {}
        # Pending one-step commitment broadcast to emit before gograb starts
        self._broadcast_target: Optional[str] = None

        # ── Metrics tracking ─────────────────────────────────────────────────
        self.metrics = {
            "duplicate_effort_attempts": 0,
            "false_plan_count":          0,
            "comm_sent":                 0,
            "comm_voi_total":            0.0,
            "belief_entropy_log":        [],
        }

    # ── LLM wrapper ───────────────────────────────────────────────────────────

    def _llm(self, prompt: str) -> str:
        """Call Anthropic claude-sonnet via the messages API."""
        if not self._api_key:
            # No key: return a deterministic fallback for testing
            return "A."
        self._llm_calls += 1
        headers = {
            "x-api-key":         self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        }
        body = {
            "model":      LLM_MODEL,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": LLM_TEMPERATURE,
            "messages":   [{"role": "user", "content": prompt}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, obs: dict, containers_name: List[str],
              goal_objects_name: List[str], rooms_name: List[str],
              room_info: List[dict], goal: dict) -> None:

        self.steps            = 0
        self.plan             = None
        self.stuck            = 0
        self.last_room        = None
        self.last_action      = None
        self.opened_containers  = set()
        self.committed_targets  = {}
        self._broadcast_target  = None
        self.containers_name  = containers_name
        self.goal_objects_name = goal_objects_name
        self.rooms_name       = rooms_name
        self.roomname2id      = {x["class_name"]: x["id"] for x in room_info}
        self.id2node          = {x["id"]: x for x in obs["nodes"]}
        self.unsatisfied      = {k: v[0] for k, v in goal.items()}
        self.satisfied        = []
        self.dialogue_history = []
        self.opponent_grabbed = []

        # ── Parse goal location from goal spec ──────────────────────────────────
        first_key  = list(goal.keys())[0]
        # Goal spec formats:
        #   Mock env: "on_cupcake_coffeetable", "inside_fork_dishwasher"
        #   Real env: "inside_cutleryfork_104" (last part is numeric node ID)
        parts      = first_key.split("_")
        last_part  = parts[-1]

        self.goal_location_id = -1
        self.goal_location    = last_part

        if last_part.isdigit():
            # Real env: last part IS the numeric node ID
            self.goal_location_id = int(last_part)
            # Find class name from obs nodes
            loc_class = ""
            for node in obs["nodes"]:
                if node["id"] == self.goal_location_id:
                    loc_class = node.get("class_name", "")
                    self.goal_location = f"<{loc_class}> ({self.goal_location_id})"
                    break
            if not loc_class:
                # Not in partial obs yet — mark for lazy resolution
                loc_class = f"node_{self.goal_location_id}"
                self.goal_location = f"<node_{self.goal_location_id}> ({self.goal_location_id})"
        else:
            # Mock env: last part is class name
            loc_class = last_part
            # Search obs nodes
            for node in obs["nodes"]:
                if node["class_name"] == loc_class:
                    self.goal_location_id = node["id"]
                    self.goal_location    = f"<{loc_class}> ({node['id']})"
                    break
            # Search id2node
            if self.goal_location_id == -1:
                for nid, node in self.id2node.items():
                    if node["class_name"] == loc_class:
                        self.goal_location_id = nid
                        self.goal_location    = f"<{loc_class}> ({nid})"
                        break

        self._goal_loc_class = loc_class

        # id_inside_room: maps node_id -> room_name or list of candidate rooms
        # Goal location starts as unknown (all rooms possible) if not yet resolved
        self.id_inside_room   = {
            self.opponent_agent_id: None,
        }
        if self.goal_location_id != -1:
            self.id_inside_room[self.goal_location_id] = list(rooms_name)
        self.unchecked_containers = {r: None for r in rooms_name}
        self.ungrabbed_objects    = {r: None for r in rooms_name}
        self.exhausted_rooms: set = set()
        self.sweep_mode:      bool = False
        self.sweep_queue:     list = []

        # ── Find starting room from edges (must precede partition logic) ────
        for e in obs["edges"]:
            if e["from_id"] == self.agent_id and e["relation_type"] == "INSIDE":
                self.current_room = self.id2node.get(e["to_id"])
                break

        # ── Assign search partitions based on starting room ───────────────
        # C-WAH topology: kitchen↔livingroom and bedroom↔bathroom are adjacent.
        # Each agent searches its own cluster first so the two agents split the
        # house without explicit coordination at episode start.
        _kitchen_cluster = {"kitchen", "livingroom"}
        _bedroom_cluster = {"bedroom", "bathroom"}
        _start = self.current_room["class_name"] if self.current_room else ""

        if _start in _kitchen_cluster:
            self.role                = "RETRIEVER"
            self.search_partition    = [r for r in rooms_name if r in _kitchen_cluster]
            self.secondary_partition = [r for r in rooms_name if r not in _kitchen_cluster]
        elif _start in _bedroom_cluster:
            self.role                = "DEPOSITOR"
            self.search_partition    = [r for r in rooms_name if r in _bedroom_cluster]
            self.secondary_partition = [r for r in rooms_name if r not in _bedroom_cluster]
        else:
            # Non-standard room layout — fall back to agent-ID-based halves
            _mid = len(rooms_name) // 2
            self.role                = "RETRIEVER"
            self.search_partition    = list(
                rooms_name[:_mid] if self.agent_id == 1 else rooms_name[_mid:])
            self.secondary_partition = list(
                rooms_name[_mid:] if self.agent_id == 1 else rooms_name[:_mid])

        self._start_room = (self.search_partition[0]
                            if self.search_partition else (rooms_name[0] if rooms_name else ""))

        # ── Reset per-episode broadcast state ────────────────────────────────
        self._prev_satisfied      = []
        self._broadcast_exhausted = set()
        self._broadcast_grab      = set()

        self.action_history = [
            f"[goexplore] <{self.current_room['class_name']}> "
            f"({self.current_room['id']})"
        ] if self.current_room else []

        # ── Initialise BDCA modules ──────────────────────────────────────────
        # Parse goal objects from goal spec
        obj_list = []
        for pred in goal:
            parts = pred.split("_")
            if len(parts) >= 2:
                obj_class = parts[1]
                if obj_class not in obj_list:
                    obj_list.append(obj_class)

        if USE_BELIEF_GRAPH:
            self.belief = BeliefGraph(
                agent_id     = self.agent_name,
                goal_objects = obj_list,
                teammate_id  = self.opponent_name,
            )
            self.belief.initialize_priors(
                rooms         = list(rooms_name),
                containers    = list(containers_name),
                goal_location = self.goal_location,
            )
        else:
            self.belief = None

        if USE_PARTNER_MODEL:
            self.partner_model = PartnerIntentModel(
                agent_id    = self.agent_name,
                teammate_id = self.opponent_name,
                goal_objects= obj_list,
                llm_fn      = None,
            )
        else:
            self.partner_model = None

        # Parse goal location class name from goal spec
        goal_loc_class = self.goal_location.split("<")[-1].split(">")[0] if "<" in self.goal_location else self.goal_location.split()[0]
        self.planner = BeliefAwarePlanner(
            agent_id            = self.agent_name,
            goal_objects        = obj_list,
            rooms               = list(rooms_name),
            llm_fn              = self._llm,
            goal_location_class = goal_loc_class,
        )

        self.comm_policy = (
            VOICommunicationPolicy(
                agent_id    = self.agent_name,
                teammate_id = self.opponent_name,
                goal_objects= obj_list,
            ) if USE_VOI_COMM else None
        )

        self.metrics = {
            "duplicate_effort_attempts": 0,
            "false_plan_count":          0,
            "comm_sent":                 0,
            "comm_voi_total":            0.0,
            "belief_entropy_log":        [],
        }

    # ── Observation parser ────────────────────────────────────────────────────

    def _parse_observation(self, observation: dict, goal: dict) -> dict:
        """
        Convert CoELA-format graph observation to BDCA structured dict.
        """
        obs = observation
        id2node = {x["id"]: x for x in obs["nodes"]}

        current_room      = None
        held_objects      = []
        opp_held          = []
        visible_objects   = []
        visible_containers= []
        reachable         = []
        close_set         = set()

        for e in obs["edges"]:
            x, r, y = e["from_id"], e["relation_type"], e["to_id"]
            if x == self.agent_id:
                if r == "INSIDE":
                    current_room = id2node.get(y)
                elif r in ("HOLDS_RH", "HOLDS_LH"):
                    held_objects.append(y)
                elif r == "CLOSE":
                    close_set.add(y)
                    yn = id2node.get(y)
                    if yn:
                        reachable.append(
                            f"<{yn['class_name']}> ({yn['id']})")
            elif x == self.opponent_agent_id and r in ("HOLDS_RH","HOLDS_LH"):
                opp_held.append(id2node[y])

        # Build visible object/container lists from current room
        if current_room:
            cr_id = current_room["id"]
            for e in obs["edges"]:
                if e["to_id"] == cr_id and e["relation_type"] == "INSIDE":
                    nid  = e["from_id"]
                    node = id2node.get(nid)
                    if node is None:
                        continue
                    if "GRABBABLE" in node.get("properties", []):
                        visible_objects.append({
                            "id":         nid,
                            "class_name": node["class_name"],
                            "state":      node.get("states", []),
                        })
                    if "CONTAINERS" in node.get("properties", []):
                        visible_containers.append({
                            "id":         nid,
                            "class_name": node["class_name"],
                            "state":      "OPEN" if "OPEN" in node.get("states",[])
                                          else "CLOSED",
                        })

        # Teammate visibility
        tm_visible = False
        tm_room    = None
        for node in obs["nodes"]:
            if node["id"] == self.opponent_agent_id:
                for e in obs["edges"]:
                    if (e["from_id"] == self.opponent_agent_id
                            and e["relation_type"] == "INSIDE"):
                        tm_node = id2node.get(e["to_id"])
                        if tm_node:
                            tm_room    = tm_node["class_name"]
                            tm_visible = (current_room is not None
                                          and tm_room == current_room["class_name"])

        # Completed subgoals from goal spec
        completed = []
        for predicate, (count, _, _) in goal.items():
            parts    = predicate.split("_")
            obj_class= parts[1]
            found    = sum(1 for s in self.satisfied
                          if s["class_name"] == obj_class)
            if found >= count:
                completed.append(obj_class)

        return {
            "current_room":       current_room["class_name"] if current_room else "unknown",
            "current_room_id":    current_room["id"] if current_room else -1,
            "visible_objects":    visible_objects,
            "visible_containers": visible_containers,
            # Open containers seen this step -> mark as inspected in belief graph
            "inspected_containers": [
                {"id": str(c["id"]),
                 "class_name": c["class_name"],
                 "contents": [obj["class_name"] for obj in visible_objects]}
                for c in visible_containers if c["state"] == "OPEN"
            ],
            "held_objects":       [id2node[o]["class_name"]
                                   for o in held_objects if o in id2node],
            "held_object_ids":    held_objects,
            "teammate_visible":   tm_visible,
            "teammate_id":        self.opponent_name,
            "teammate_room":      tm_room or "unknown",
            "teammate_holding":   [o["class_name"] for o in opp_held],
            "completed_subgoals": completed,
            "reachable_objects":  reachable,
            "goal_location":      self.goal_location,
            "room_ids":           self.roomname2id,
            # Objects seen in previous room visits — key addition for cross-room planning
            "ungrabbed_by_room":  {r: v for r, v in self.ungrabbed_objects.items() if v},
            "explored_rooms":     [r for r, v in self.ungrabbed_objects.items() if v is not None],
            "exhausted_rooms":    list(self.exhausted_rooms),
            "search_partition":   getattr(self, "search_partition", self.rooms_name),
            # All unchecked containers across all visited rooms
            "all_unchecked_containers": [
                c for room_list in self.unchecked_containers.values()
                if room_list for c in room_list
            ],
            # Object IDs committed by either agent — planner excludes these
            "committed_targets": {
                oid: exp for oid, exp in self.committed_targets.items()
                if exp > self.steps
            },
        }

    # ── Low-level action execution (CoELA-compatible skill book) ─────────────

    def goexplore(self) -> Optional[str]:
        target_room_id = int(self.plan.split(" ")[-1][1:-1])
        if self.current_room["id"] == target_room_id:
            self.plan = None
            return None
        return self.plan.replace("[goexplore]", "[walktowards]")

    def gocheck(self) -> Optional[str]:
        assert len(self.grabbed_objects) < 2
        parts = self.plan.split(" ")
        try:
            target_id   = int(parts[-1][1:-1])
            target_name = parts[1]
        except (ValueError, IndexError):
            self.plan = None
            return None

        target_room = self.id_inside_room.get(target_id,
                                               self.current_room["class_name"]
                                               if self.current_room else "kitchen")
        if isinstance(target_room, list):
            target_room = target_room[0] if target_room else "kitchen"

        if self.current_room and self.current_room["class_name"] != target_room:
            return (f"[walktowards] <{target_room}> "
                    f"({self.roomname2id.get(target_room, 0)})")

        target_node = self.id2node.get(target_id)
        already_open_node = target_node and "OPEN" in target_node.get("states", [])
        if already_open_node or target_id in self.opened_containers:
            # Container already open — mark inspected and clear plan
            if self.belief:
                self.belief.mark_inspected(str(target_id), target_name.strip("<>"))
            self.plan = None
            return None

        reach_str = f"{target_name} ({target_id})"
        # Remove angle brackets for comparison
        reach_clean = [r.replace("<","").replace(">","") for r in self.reachable_objects]
        clean_str   = f"{target_name.replace('<','').replace('>','')} ({target_id})"
        if clean_str in reach_clean or reach_str in self.reachable_objects:
            # Optimistically record open so next step doesn't re-issue [open]
            self.opened_containers.add(target_id)
            if target_id in self.id2node:
                _n = dict(self.id2node[target_id])
                _n["states"] = [s for s in _n.get("states", []) if s != "CLOSED"]
                if "OPEN" not in _n["states"]:
                    _n["states"] = _n["states"] + ["OPEN"]
                self.id2node[target_id] = _n
            return self.plan.replace("[gocheck]", "[open]")
        # Walk towards the container
        return self.plan.replace("[gocheck]", "[walktowards]")

    def gograb(self) -> Optional[str]:
        target_id   = int(self.plan.split(" ")[-1][1:-1])
        target_name = self.plan.split(" ")[1]
        if target_id in self.grabbed_objects:
            self.plan = None
            return None
        assert len(self.grabbed_objects) < 2
        target_room = self.id_inside_room.get(target_id,
                                               self.current_room["class_name"])
        if isinstance(target_room, list):
            target_room = target_room[0] if target_room else "unknown"
        if self.current_room["class_name"] != target_room:
            return (f"[walktowards] <{target_room}> "
                    f"({self.roomname2id.get(target_room, 0)})")
        # Check still available
        ungrabbed_in_room = self.ungrabbed_objects.get(
            self.current_room["class_name"], []) or []
        if target_id not in self.id2node or \
           target_id not in [w["id"] for w in ungrabbed_in_room] or \
           target_id in [x["id"] for x in self.opponent_grabbed]:
            self.plan = None
            self.metrics["false_plan_count"] += 1
            return None
        if f"{target_name} ({target_id})" in self.reachable_objects:
            return self.plan.replace("[gograb]", "[grab]")
        return self.plan.replace("[gograb]", "[walktowards]")

    def goput(self) -> Optional[str]:
        if not self.grabbed_objects:
            self.plan = None
            return None

        gl_id   = self.goal_location_id
        held_id = self.grabbed_objects[0]

        # ── Find goal room by scanning ALL ever-seen nodes ────────────────
        target_room = None

        # First check id_inside_room cache
        room_val = self.id_inside_room.get(gl_id)
        if isinstance(room_val, str) and room_val in self.rooms_name:
            target_room = room_val
        elif isinstance(room_val, list) and len(room_val) == 1:
            target_room = room_val[0]

        # Second: scan every node we have ever placed in id_inside_room
        if target_room is None:
            for nid, room in self.id_inside_room.items():
                if nid == gl_id and isinstance(room, str) and room in self.rooms_name:
                    target_room = room
                    break

        # Third: check current obs directly
        if target_room is None:
            loc_class = getattr(self, "_goal_loc_class", "")
            for nid, node in self.id2node.items():
                if node.get("class_name") == loc_class or nid == gl_id:
                    # Try to find this node's room
                    r = self.id_inside_room.get(nid)
                    if isinstance(r, str) and r in self.rooms_name:
                        target_room = r
                        self.id_inside_room[gl_id] = r
                        break

        # Fallback: use first room in list or first unexhausted room
        if target_room is None:
            if isinstance(room_val, list) and room_val:
                target_room = room_val[0]
            else:
                for r in self.rooms_name:
                    if r not in self.exhausted_rooms:
                        target_room = r
                        break
                else:
                    target_room = self.rooms_name[0] if self.rooms_name else "kitchen"

        # ── Walk to goal room ─────────────────────────────────────────────
        current = self.current_room["class_name"] if self.current_room else ""
        if current != target_room:
            room_id = self.roomname2id.get(target_room, 0)
            return f"[walktowards] <{target_room}> ({room_id})"

        # ── In correct room: execute put immediately ──────────────────────
        # Final resolution of gl_id — scan id2node one more time
        if gl_id == -1 or gl_id not in self.id2node:
            loc_class = getattr(self, "_goal_loc_class", "")
            for nid, node in self.id2node.items():
                if node.get("class_name") == loc_class:
                    gl_id = nid
                    self.goal_location_id = nid
                    self.goal_location = f"<{loc_class}> ({nid})"
                    self.id_inside_room[nid] = (
                        self.current_room["class_name"]
                        if self.current_room else target_room)
                    break

        gl_node = self.id2node.get(gl_id, {})
        held_n  = self.id2node.get(held_id, {})

        if not gl_node:
            loc_class = getattr(self, "_goal_loc_class", "")
            gl_node = {"class_name": loc_class, "id": gl_id,
                       "properties": ["SURFACE"], "states": []}
        if not held_n:
            held_n = {"class_name": "object", "id": held_id}

        # Choose action
        gl_class = gl_node.get("class_name", getattr(self, "_goal_loc_class", ""))
        is_container = ("CONTAINERS" in gl_node.get("properties", [])
                        or gl_class in self.containers_name)
        if is_container:
            # Prefer live id2node state (updated each step + optimistic overrides)
            # over gl_node which was snapshotted earlier and may be stale.
            live_node = self.id2node.get(gl_id, gl_node)
            state = live_node.get("states", [])
            already_open = (
                gl_id in self.opened_containers or
                (self.belief and (
                    self.belief.is_inspected(str(gl_id)) or
                    self.belief.is_inspected(gl_class)
                ))
            )
            if "CLOSED" in state and not already_open:
                # Optimistically record open so next step doesn't re-issue [open]
                self.opened_containers.add(gl_id)
                if gl_id in self.id2node:
                    _n = dict(self.id2node[gl_id])
                    _n["states"] = [s for s in _n.get("states", []) if s != "CLOSED"]
                    if "OPEN" not in _n["states"]:
                        _n["states"] = _n["states"] + ["OPEN"]
                    self.id2node[gl_id] = _n
                return f"[open] <{gl_class}> ({gl_id})"
            action = "[putin]"
        else:
            action = "[putback]"

        self.plan = None
        return (f"{action} <{held_n.get('class_name','object')}> ({held_id}) "
                f"<{gl_node.get('class_name','?')}> ({gl_id})")


    def check_progress(self, state: dict, goal_spec: dict):
        unsatisfied = {}
        satisfied   = []
        id2node     = {n["id"]: n for n in state["nodes"]}
        for key, value in goal_spec.items():
            parts   = key.split("_")
            rel     = parts[0].upper()
            obj_cls = parts[1]
            cnt     = value[0]
            for edge in state["edges"]:
                if cnt == 0:
                    break
                if (edge["relation_type"] == rel
                        and edge["to_id"] == self.goal_location_id
                        and id2node.get(edge["from_id"],{}).get("class_name") == obj_cls):
                    satisfied.append(id2node[edge["from_id"]])
                    cnt -= 1
            if cnt > 0:
                unsatisfied[key] = cnt
        return satisfied, unsatisfied

    def filter_graph(self, obs: dict) -> dict:
        loc_class = getattr(self, "_goal_loc_class", "")
        all_rel = (self.containers_name + self.goal_objects_name
                   + self.rooms_name + ["character"])
        if loc_class and loc_class not in all_rel:
            all_rel = all_rel + [loc_class]

        rel_ids = [n["id"] for n in obs["nodes"]
                   if n["class_name"] in all_rel]
        rel_ids = [x for x in rel_ids
                   if all(x != y["id"] for y in self.satisfied)]

        # Always include goal location even if not in all_rel
        if self.goal_location_id != -1:
            if any(n["id"] == self.goal_location_id for n in obs["nodes"]):
                if self.goal_location_id not in rel_ids:
                    rel_ids.append(self.goal_location_id)

        return {
            "edges": [e for e in obs["edges"]
                      if e["from_id"] in rel_ids and e["to_id"] in rel_ids],
            "nodes": [n for n in obs["nodes"] if n["id"] in rel_ids],
        }


    def get_action(self, observation: dict, goal: dict
                   ) -> Tuple[Optional[str], dict]:

        # ── 0a. Lazy goal location resolution ────────────────────────────────
        # Purge expired commitment entries each step
        if self.committed_targets:
            self.committed_targets = {
                k: v for k, v in self.committed_targets.items()
                if v > self.steps
            }

        # If goal_location_id is still -1, scan this observation for it
        if self.goal_location_id == -1:
            loc_class = getattr(self, "_goal_loc_class", "")
            for node in observation.get("nodes", []):
                if node.get("class_name") == loc_class:
                    self.goal_location_id = node["id"]
                    self.goal_location    = f"<{loc_class}> ({node['id']})"
                    self.id_inside_room[self.goal_location_id] = list(
                        self.rooms_name)
                    break

        # ── 0b. Absorb messages ───────────────────────────────────────────────
        for i, msg in enumerate(observation.get("messages", [None, None])):
            if msg is not None:
                sender = "Alice" if i == 0 else "Bob"
                self.dialogue_history.append(f"{sender}: {msg}")
                parsed_msg = self._parse_message(msg, sender)
                if parsed_msg:
                    msg_type = parsed_msg.get("type")
                    if msg_type == "commitment":
                        oid    = parsed_msg["object_id"]
                        oclass = parsed_msg["object_class"]
                        self.committed_targets[oid] = (
                            self.steps + COMMITMENT_DURATION)
                        if self.partner_model:
                            self.partner_model.force_collision_risk(
                                oclass, self.steps + COMMITMENT_DURATION)
                    elif msg_type == "room_exhausted":
                        room = parsed_msg["room"]
                        # Add to own exhausted set so planner skips this room
                        self.exhausted_rooms.add(room)
                        # Zero out belief mass for all goal objects in this room
                        if self.belief:
                            for obj in self.goal_objects_name:
                                self.belief._update_location(
                                    obj, room, self.steps, "inferred", 0.01)
                    if self.belief:
                        self.belief.update_from_message(parsed_msg, self.steps)

        # ── 1. Check progress ────────────────────────────────────────────────
        satisfied, unsatisfied = self.check_progress(observation, goal)
        if satisfied:
            prev_satisfied = len(self.satisfied)
            self.satisfied   = satisfied
            self.unsatisfied = unsatisfied

            # If new subgoal completed and more of same class needed,
            # boost container belief for remaining instances
            if self.belief and len(satisfied) > prev_satisfied:
                for pred, count in self.unsatisfied.items():
                    if count > 0:
                        obj_class = pred.split("_")[1]
                        # Boost all uninspected containers for this class
                        for cname in self.containers_name:
                            if not self.belief.is_inspected(cname):
                                cur = self.belief.get_confidence(
                                    obj_class, "location", cname)
                                self.belief._update_location(
                                    obj_class, cname,
                                    self.steps, "inferred",
                                    min(0.85, cur * 2.0 + 0.15))

        obs_graph = self.filter_graph(observation)
        self.grabbed_objects   = []
        self.opponent_grabbed  = []
        self.reachable_objects = []
        self.id2node = {x["id"]: x for x in obs_graph["nodes"]}

        # Re-apply opened_containers overrides: a stale partial obs may still
        # report a container as CLOSED even after we successfully opened it.
        for _oid in self.opened_containers:
            if _oid in self.id2node and "CLOSED" in self.id2node[_oid].get("states", []):
                _n = dict(self.id2node[_oid])
                _n["states"] = [s for s in _n["states"] if s != "CLOSED"]
                if "OPEN" not in _n["states"]:
                    _n["states"] = _n["states"] + ["OPEN"]
                self.id2node[_oid] = _n

        for e in obs_graph["edges"]:
            x, r, y = e["from_id"], e["relation_type"], e["to_id"]
            if x == self.agent_id:
                if r == "INSIDE":
                    self.current_room = self.id2node.get(y)
                elif r in ("HOLDS_RH", "HOLDS_LH"):
                    self.grabbed_objects.append(y)
                elif r == "CLOSE":
                    yn = self.id2node.get(y)
                    if yn:
                        self.reachable_objects.append(
                            f"<{yn['class_name']}> ({yn['id']})")
            elif (x == self.opponent_agent_id
                  and r in ("HOLDS_RH", "HOLDS_LH")):
                self.opponent_grabbed.append(self.id2node[y])

        unchecked = []
        ungrabbed = []
        for xn in obs_graph["nodes"]:
            if (xn["id"] in self.grabbed_objects
                    or xn["id"] in [w["id"] for w in self.opponent_grabbed]):
                for room, ul in self.ungrabbed_objects.items():
                    if ul is None:
                        continue
                    ul[:] = [u for u in ul if u["id"] != xn["id"]]
                continue
            self.id_inside_room[xn["id"]] = (
                self.current_room["class_name"]
                if self.current_room else "unknown")

            # If this is the goal location, resolve its room definitively
            if xn["id"] == self.goal_location_id and self.current_room:
                self.id_inside_room[self.goal_location_id] = (
                    self.current_room["class_name"])
                # Track if goal container is open
                if "OPEN" in xn.get("states", []) and self.belief:
                    self.belief.mark_inspected(
                        str(self.goal_location_id),
                        xn.get("class_name", ""))

            if (xn["class_name"] in self.containers_name
                    and "CLOSED" in xn.get("states", [])
                    and xn["id"] != self.goal_location_id):
                xn_with_room = dict(xn)
                xn_with_room["room"] = (self.current_room["class_name"]
                                        if self.current_room else "unknown")
                unchecked.append(xn_with_room)
            if (any(xn["class_name"] == g.split("_")[1]
                    for g in self.unsatisfied)
                    and all(xn["id"] != y["id"] for y in self.satisfied)
                    and "GRABBABLE" in xn.get("properties", [])
                    and xn["id"] not in self.grabbed_objects
                    and xn["id"] not in [w["id"] for w in self.opponent_grabbed]):
                ungrabbed.append(xn)
                # Boost belief: if we see a goal object here, same class
                # objects are more likely nearby — boost current room AND its containers
                if self.belief and self.current_room:
                    obj_class = xn["class_name"]
                    cr = self.current_room["class_name"]
                    # Boost room probability strongly
                    self.belief._update_location(
                        obj_class, cr, self.steps, "inferred", 0.75)
                    # Boost containers in this room
                    for cont_name in self.containers_name:
                        cont_room = None
                        for nid, node in self.id2node.items():
                            if node.get("class_name") == cont_name:
                                cont_room = self.id_inside_room.get(nid)
                                if isinstance(cont_room, str) and cont_room == cr:
                                    old_c = self.belief.get_confidence(
                                        obj_class, "location", cont_name)
                                    self.belief._update_location(
                                        obj_class, cont_name,
                                        self.steps, "inferred",
                                        min(0.85, old_c + 0.25))
                                    break

        if self.current_room:
            cr = self.current_room["class_name"]
            self.unchecked_containers[cr] = unchecked
            self.ungrabbed_objects[cr]    = ungrabbed

            # If goal location not seen in this room, remove it from candidates
            if (self.goal_location_id != -1
                    and isinstance(self.id_inside_room.get(self.goal_location_id), list)
                    and self.goal_location_id not in [xn["id"] for xn in obs_graph["nodes"]]
                    and cr in self.id_inside_room[self.goal_location_id]):
                self.id_inside_room[self.goal_location_id].remove(cr)
                # If only one room left, resolve definitively
                remaining = self.id_inside_room[self.goal_location_id]
                if len(remaining) == 1:
                    self.id_inside_room[self.goal_location_id] = remaining[0]

            # Mark room as exhausted if no unchecked containers and no ungrabbed objects
            if (not unchecked and not ungrabbed
                    and self.ungrabbed_objects.get(cr) is not None):
                self.exhausted_rooms.add(cr)
            # If we are in this room and goal objects are NOT directly visible,
            # redistribute their belief mass away from this room towards containers.
            if self.belief:
                incomplete_now = [
                    k.split("_")[1] for k in self.unsatisfied
                    if self.unsatisfied.get(k, 0) > 0
                ]
                visible_classes = {xn["class_name"] for xn in obs_graph["nodes"]
                                   if "GRABBABLE" in xn.get("properties", [])}
                for obj in incomplete_now:
                    if obj not in visible_classes:
                        # Object not directly visible in current room —
                        # reduce room probability, boost uninspected containers
                        cur_conf = self.belief.get_confidence(obj, "location", cr)
                        if cur_conf > 0.15:
                            # Redistribute: split half to uninspected containers in room
                            uninspected = [
                                c for c in self.containers_name
                                if (not self.belief.is_inspected(c)
                                    and c not in [str(self.goal_location_id)])
                            ]
                            if uninspected and unchecked:
                                # Boost containers in this room
                                boost_per = cur_conf * 0.4 / len(unchecked)
                                for cont in unchecked:
                                    cname = cont["class_name"]
                                    old_c = self.belief.get_confidence(
                                        obj, "location", cname)
                                    self.belief._update_location(
                                        obj, cname,
                                        self.steps,
                                        "inferred",
                                        min(0.85, old_c + boost_per)
                                    )
                            # Reduce direct room probability
                            self.belief._update_location(
                                obj, cr, self.steps, "inferred",
                                max(0.03, cur_conf * 0.3)
                            )

        # Track opponent room
        if (self.current_room and
                self.id_inside_room.get(self.opponent_agent_id)
                == self.current_room["class_name"]):
            self.opponent_grabbed = self.opponent_grabbed[:]

        # ── 2. Build structured obs for BDCA modules ─────────────────────────
        obs_parsed = self._parse_observation(observation, goal)

        # ── 3. Update belief graph ────────────────────────────────────────────
        if self.belief:
            self.belief.update_from_observation(obs_parsed, self.steps)
            if USE_BELIEF_DECAY:
                tm_room = self.id_inside_room.get(self.opponent_agent_id)
                if isinstance(tm_room, list):
                    tm_room = tm_room[0] if tm_room else None
                self.belief.apply_decay(self.steps, tm_room)
            self.metrics["belief_entropy_log"].append(
                self.belief.total_entropy())

        # ── 4. Partner prediction ─────────────────────────────────────────────
        incomplete_goals = [
            k.split("_")[1] for k in self.unsatisfied
            if self.unsatisfied[k] > 0
        ]
        partner_pred = None
        if self.partner_model and self.belief:
            partner_pred = self.partner_model.predict(
                self.belief,
                [{"content": m} for m in self.dialogue_history[-5:]],
                self.steps,
                incomplete_goals,
            )

        # Track duplicate effort — only count when BOTH agents
        # actually attempt the same action on the same step
        # (counted in runner.py from ground truth, not here)
        pass  # duplicate effort tracked in runner via check_progress

        # ── 5. VOI communication ──────────────────────────────────────────────
        comm_decision  = None
        just_completed = None
        just_exhausted = None

        # Subgoal completions
        if len(satisfied) > len(self._prev_satisfied):
            new = [s for s in satisfied if s not in self._prev_satisfied]
            if new:
                just_completed = new[0]["class_name"]
        self._prev_satisfied = list(satisfied)

        # Process outgoing commitment messages immediately
        # so our own committed_targets blocks us from double-committing
        if comm_decision and isinstance(comm_decision, str):
            if comm_decision.startswith("[send_message]"):
                msg = comm_decision.split("<")[-1].rstrip(">")
                if msg.startswith("targeting:"):
                    parts = msg.split(":")
                    if len(parts) >= 3:
                        oid = parts[2]
                        self.committed_targets[oid] = self.steps + 10

        # Room-exhaustion broadcasts (highest priority after commitment grabs)
        # Uses structured "exhausted:<room>" format so partner can parse it exactly.
        newly_exhausted = self.exhausted_rooms - self._broadcast_exhausted
        if newly_exhausted:
            self._broadcast_exhausted.update(newly_exhausted)
            just_exhausted = list(newly_exhausted)[0]  # plain room name

        # Force broadcast when first picking up a goal object
        new_grabs = set(self.grabbed_objects) - self._broadcast_grab
        if new_grabs:
            self._broadcast_grab.update(new_grabs)
            held_class = self.id2node.get(list(new_grabs)[0], {}).get("class_name", "")
            cr = self.current_room["class_name"] if self.current_room else ""
            just_completed = f"found:{held_class}:{cr}"

        if self.comm_policy and partner_pred:
            comm_decision = self.comm_policy.decide(
                self.belief, partner_pred, self.steps,
                incomplete_goals, just_completed,
                obs_parsed["current_room"],
                just_exhausted=just_exhausted,
            )
            if comm_decision.should_send:
                self.metrics["comm_sent"]     += 1
                self.metrics["comm_voi_total"] += comm_decision.voi

        # ── 6. Plan via BDCA planner ──────────────────────────────────────────
        info = {
            "obs": obs_parsed,
            "graph": obs_graph,
            "steps": self.steps,
        }

        action     = None
        lm_retries = 0
        while action is None:
            if self.plan is None:
                # Hard fallback after too many retries
                if lm_retries > 8:
                    gl_room = self.id_inside_room.get(self.goal_location_id)
                    if isinstance(gl_room, list):
                        gl_room = gl_room[0] if gl_room else (self.rooms_name[0] if self.rooms_name else "kitchen")
                    action = (f"[walktowards] <{gl_room}> ({self.roomname2id.get(gl_room,0)})"
                              if gl_room and gl_room != (self.current_room["class_name"] if self.current_room else "")
                              else "[wait]")
                    break

                if not incomplete_goals and not self.grabbed_objects:
                    self.plan = "[wait]"
                elif len(self.grabbed_objects) >= MAX_HELD:
                    self.plan = f"[goput] {self.goal_location}"
                elif self.planner and self.belief and partner_pred:
                    self.plan, plan_info = self.planner.plan(
                        self.belief, partner_pred, obs_parsed,
                        incomplete_goals, self.action_history,
                        self.steps, comm_decision,
                    )
                    info.update({"BDCA_plan": plan_info})
                    # If planner chose to grab an object, register a commitment
                    # broadcast so the partner stops targeting the same object
                    if self.plan.startswith("[gograb]") and not self._broadcast_target:
                        _parts = self.plan.split(" ")
                        if len(_parts) >= 3:
                            _oclass  = _parts[1].strip("<>")
                            _oid_str = _parts[2].strip("()")
                            self.committed_targets[_oid_str] = (
                                self.steps + COMMITMENT_DURATION)
                            if self.comm_policy:
                                _cdec = self.comm_policy.decide(
                                    self.belief, partner_pred, self.steps,
                                    incomplete_goals,
                                    my_room=obs_parsed.get("current_room", ""),
                                    just_targeted=f"{_oclass}:{_oid_str}",
                                )
                                self._broadcast_target = (
                                    f"[send_message] <{_cdec.content}>")
                            else:
                                self._broadcast_target = (
                                    f"[send_message]"
                                    f" <targeting:{_oclass}:{_oid_str}>")
                    # If planner returned same-room explore or wait, pick best different room
                    cr = self.current_room["class_name"] if self.current_room else ""
                    if incomplete_goals and (
                        self.plan == "[wait]" or
                        (self.plan.startswith("[goexplore]") and f"<{cr}>" in self.plan)
                    ):
                        best_room = cr
                        best_mass = -1.0
                        for r in self.rooms_name:
                            if r == cr:
                                continue
                            mass = sum(
                                self.belief.get_confidence(obj, "location", r)
                                for obj in incomplete_goals
                            )
                            if mass > best_mass:
                                best_mass = mass
                                best_room = r
                        if best_room != cr:
                            rid = self.roomname2id.get(best_room, 0)
                            self.plan = f"[goexplore] <{best_room}> ({rid})"
                        else:
                            # All mass in current room — check containers or grab
                            self.plan = "[wait]"
                else:
                    # Modules not ready — explore first different room
                    cr = self.current_room["class_name"] if self.current_room else ""
                    for r in self.rooms_name:
                        if r != cr:
                            self.plan = f"[goexplore] <{r}> ({self.roomname2id.get(r,0)})"
                            break
                    else:
                        self.plan = "[wait]"

                self.action_history.append(
                    "[send_message]" if self.plan.startswith("[send_message]")
                    else self.plan
                )
                lm_retries += 1

            # Execute current plan
            if self.plan.startswith("[goexplore]"):
                action = self.goexplore()
            elif self.plan.startswith("[gocheck]"):
                action = self.gocheck()
                # gocheck returns None when container already open — re-plan
                if action is None and self.plan is None:
                    continue  # plan was cleared, will re-plan next iteration
                elif action is None:
                    self.plan = None  # force re-plan
            elif self.plan.startswith("[gograb]"):
                if self._broadcast_target:
                    # First step of a new gograb: emit commitment broadcast.
                    # Plan stays as gograb — physical grab resumes next step.
                    action = self._broadcast_target
                    self._broadcast_target = None
                    self.metrics["comm_sent"]      += 1
                    self.metrics["comm_voi_total"] += 999.0
                else:
                    action = self.gograb()
            elif self.plan.startswith("[goput]"):
                action = self.goput()
            elif self.plan.startswith("[send_message]"):
                action  = self.plan[:]
                self.plan = None
            elif self.plan.startswith("[wait]"):
                action = None
                break
            else:
                self.plan = None  # unrecognised — re-plan

        # ── 7. Stuck detection ────────────────────────────────────────────────
        if (action == self.last_action
                and self.current_room == self.last_room):
            self.stuck += 1
        else:
            self.stuck = 0

        if self.stuck > 20:
            self.plan = None
            cr        = self.current_room["class_name"] if self.current_room else ""
            gl_room   = self.id_inside_room.get(self.goal_location_id)
            if isinstance(gl_room, list):
                gl_room = gl_room[0] if gl_room else cr
            action    = (f"[walktowards] <{gl_room}> "
                         f"({self.roomname2id.get(gl_room,0)})"
                         if cr != gl_room else
                         f"[walktowards] {self.goal_location}")
            self.stuck = 0
            self.action_history[-1] += " (stuck_recovery)"

        self.last_action = action
        self.last_room   = self.current_room
        self.steps      += 1
        info.update({"plan": self.plan, "metrics": dict(self.metrics)})
        return action, info

    # ── Message parser ────────────────────────────────────────────────────────

    def _parse_message(self, msg: str, sender: str) -> Optional[dict]:
        """
        Lightweight rule-based message parser.
        Extracts structured intent from natural language messages.
        """
        # Commitment broadcast (highest priority — exact prefix match)
        if msg.startswith("targeting:"):
            parts = msg.split(":")
            if len(parts) == 3:
                return {
                    "type":         "commitment",
                    "object_class": parts[1],
                    "object_id":    parts[2],
                    "sender":       sender,
                }

        # Room-exhaustion broadcast — partner has fully searched this room
        if msg.startswith("exhausted:"):
            room = msg[len("exhausted:"):]
            if room:
                return {
                    "type":   "room_exhausted",
                    "room":   room,
                    "sender": sender,
                }

        msg_lower = msg.lower()

        # Subgoal complete
        for kw in ["completed", "finished", "done", "put", "placed"]:
            if kw in msg_lower:
                for obj in self.goal_objects_name:
                    if obj in msg_lower:
                        return {
                            "type":      "subgoal_complete",
                            "object_id": obj,
                            "sender":    sender,
                        }

        # Object found
        for obj in self.goal_objects_name:
            if obj in msg_lower:
                for room in self.rooms_name:
                    if room in msg_lower:
                        return {
                            "type":      "object_found",
                            "object_id": obj,
                            "room":      room,
                            "sender":    sender,
                        }

        # Intent (heading to room)
        for kw in ["heading", "going", "will go", "exploring"]:
            if kw in msg_lower:
                for room in self.rooms_name:
                    if room in msg_lower:
                        return {
                            "type":   "intent",
                            "room":   room,
                            "sender": sender,
                        }

        return None
