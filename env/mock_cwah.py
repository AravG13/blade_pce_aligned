"""
BDCA Mock C-WAH Environment
============================
Simulates the C-WAH graph-based observation API without requiring
the VirtualHome simulator. Lets you develop and test the full agent
pipeline on any machine before plugging in the real environment.

The observation format exactly matches CoELA's symbolic observation:
  obs = {
    "nodes": [...],   # graph nodes with id, class_name, states, properties
    "edges": [...],   # from_id, to_id, relation_type
    "messages": [msg_agent1, msg_agent2]
  }
  goal = { "on_cupcake_coffeetable": [2, True, 2], ... }

Usage:
    env = MockCWAHEnv(task_id=0, seed=42)
    obs = env.reset()
    obs, done, info = env.step({0: action_a, 1: action_b})
"""

from __future__ import annotations
import random
import copy
from typing import Dict, List, Optional, Tuple, Any


# ── Task definitions (match CoELA's C-WAH exactly) ────────────────────────────
TASKS = {
    0: {
        "name": "Prepare afternoon tea",
        "goal": {
            "on_cupcake_coffeetable":  [2, True, 2],
            "on_pudding_coffeetable":  [1, True, 1],
            "on_apple_coffeetable":    [1, True, 1],
            "on_juice_coffeetable":    [1, True, 1],
        }
    },
    1: {
        "name": "Wash dishes",
        "goal": {
            "inside_plate_dishwasher": [1, True, 1],
            "inside_fork_dishwasher":  [1, True, 1],
        }
    },
    2: {
        "name": "Put groceries",
        "goal": {
            "inside_cupcake_fridge":   [2, True, 2],
            "inside_apple_fridge":     [1, True, 1],
            "inside_juice_fridge":     [1, True, 1],
        }
    },
    3: {
        "name": "Set up dinner table",
        "goal": {
            "on_plate_dinnertable":  [1, True, 1],
            "on_fork_dinnertable":   [1, True, 1],
        }
    },
}

# ── Node / edge id counters ───────────────────────────────────────────────────
_ID_COUNTER = 100


def _new_id() -> int:
    global _ID_COUNTER
    _ID_COUNTER += 1
    return _ID_COUNTER


class MockCWAHEnv:
    """
    A self-contained mock of the C-WAH symbolic environment.

    Agents interact via graph observations. Objects can be:
    - inside containers
    - on surfaces
    - held by agents
    - in rooms

    State mutates as agents grab/put/open objects.
    """

    ROOMS = ["kitchen", "livingroom", "bedroom", "bathroom"]
    CONTAINERS = {
        "kitchen":    ["fridge", "kitchencabinet", "microwave"],
        "livingroom": ["cabinet"],
        "bedroom":    ["wardrobe"],
        "bathroom":   ["bathroomcabinet"],
    }
    SURFACES = {
        "kitchen":    ["kitchentable", "dinnertable"],
        "livingroom": ["coffeetable", "sofa"],
        "bedroom":    ["bed"],
        "bathroom":   [],
    }

    def __init__(self, task_id: int = 0, seed: int = 42,
                 add_noise_objects: int = 0):
        self.task_id = task_id % len(TASKS)
        self.seed = seed
        self.add_noise_objects = add_noise_objects
        self.rng = random.Random(seed)

        task = TASKS[self.task_id]
        self.task_name = task["name"]
        self.goal_spec_raw = task["goal"]

        # Parse goal objects and goal location
        self.goal_objects: List[str] = []
        self.goal_location_class: str = ""
        self.goal_relation: str = ""
        for predicate in self.goal_spec_raw:
            parts = predicate.split("_")
            relation, obj_class, loc_class = parts[0], parts[1], parts[2]
            if obj_class not in self.goal_objects:
                self.goal_objects.append(obj_class)
            self.goal_location_class = loc_class
            self.goal_relation = relation

        self.max_episode_length = 250
        self.step_count = 0
        self.done = False

        # Built during reset()
        self.nodes: Dict[int, dict] = {}
        self.edges: List[dict] = []
        self.agent_ids: List[int] = []
        self.goal_location_id: int = -1
        self.room_ids: Dict[str, int] = {}
        self.messages: List[Optional[str]] = [None, None]

        # Public attributes CoELA expects
        self.all_containers_name: List[str] = []
        self.all_goal_objects_name: List[str] = []
        self.all_room_name: List[str] = list(self.ROOMS)
        self.room_info: List[dict] = []
        self.goal_spec: Dict[int, dict] = {}

        self._built = False

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[int] = None) -> List[dict]:
        global _ID_COUNTER
        _ID_COUNTER = 100

        if task_id is not None:
            self.task_id = task_id % len(TASKS)
            task = TASKS[self.task_id]
            self.task_name = task["name"]
            self.goal_spec_raw = task["goal"]

        self.step_count = 0
        self.done = False
        self.messages = [None, None]
        self.nodes = {}
        self.edges = []

        self._build_world()
        self._built = True

        obs = [self._get_observation(agent_id=aid)
               for aid in self.agent_ids]
        return obs

    def _build_world(self):
        """Construct the full symbolic world graph."""
        rng = self.rng

        # ── Rooms ────────────────────────────────────────────────────────────
        for room_name in self.ROOMS:
            nid = _new_id()
            self.nodes[nid] = {
                "id": nid,
                "class_name": room_name,
                "category": "Rooms",
                "states": [],
                "properties": ["ROOM"],
            }
            self.room_ids[room_name] = nid

        # ── Agents ───────────────────────────────────────────────────────────
        self.agent_ids = []
        start_rooms = rng.sample(self.ROOMS, 2)
        for i in range(2):
            aid = i + 1
            self.agent_ids.append(aid)
            self.nodes[aid] = {
                "id": aid,
                "class_name": "character",
                "category": "Agents",
                "states": [],
                "properties": ["PERSON"],
                "is_agent": True,
            }
            room_id = self.room_ids[start_rooms[i]]
            self.edges.append({
                "from_id": aid, "to_id": room_id,
                "relation_type": "INSIDE"
            })

        # ── Containers ────────────────────────────────────────────────────────
        self.all_containers_name = []
        for room_name, container_list in self.CONTAINERS.items():
            room_id = self.room_ids[room_name]
            for cname in container_list:
                cid = _new_id()
                self.nodes[cid] = {
                    "id": cid,
                    "class_name": cname,
                    "category": "Furniture",
                    "states": ["CLOSED"],
                    "properties": ["CONTAINERS", "SURFACE"],
                }
                self.edges.append({
                    "from_id": cid, "to_id": room_id,
                    "relation_type": "INSIDE"
                })
                if cname not in self.all_containers_name:
                    self.all_containers_name.append(cname)

        # ── Surfaces (goal locations) ──────────────────────────────────────
        for room_name, surface_list in self.SURFACES.items():
            room_id = self.room_ids[room_name]
            for sname in surface_list:
                sid = _new_id()
                self.nodes[sid] = {
                    "id": sid,
                    "class_name": sname,
                    "category": "Furniture",
                    "states": [],
                    "properties": ["SURFACE"],
                }
                self.edges.append({
                    "from_id": sid, "to_id": room_id,
                    "relation_type": "INSIDE"
                })
                if sname == self.goal_location_class:
                    self.goal_location_id = sid

        # If goal location is a container (e.g. fridge, dishwasher),
        # it was already created in the containers loop above.
        # Find it now so we never create a duplicate.
        if self.goal_location_id == -1:
            for nid, node in self.nodes.items():
                if node["class_name"] == self.goal_location_class:
                    self.goal_location_id = nid
                    break

        # If goal location not placed yet, add it
        if self.goal_location_id == -1:
            kitchen_id = self.room_ids["kitchen"]
            sid = _new_id()
            self.nodes[sid] = {
                "id": sid,
                "class_name": self.goal_location_class,
                "category": "Appliances",
                "states": ["CLOSED"],
                "properties": ["CONTAINERS", "SURFACE"],
            }
            self.edges.append({
                "from_id": sid, "to_id": kitchen_id,
                "relation_type": "INSIDE"
            })
            self.goal_location_id = sid
            if self.goal_location_class not in self.all_containers_name:
                self.all_containers_name.append(self.goal_location_class)

        # ── Goal objects ───────────────────────────────────────────────────
        self.all_goal_objects_name = list(self.goal_objects)
        placed_objects: Dict[str, List[int]] = {
            obj: [] for obj in self.goal_objects}

        for predicate, (count, _, _) in self.goal_spec_raw.items():
            parts = predicate.split("_")
            obj_class = parts[1]
            for _ in range(count):
                oid = _new_id()
                self.nodes[oid] = {
                    "id": oid,
                    "class_name": obj_class,
                    "category": "Props",
                    "states": ["GRABBABLE"],
                    "properties": ["GRABBABLE"],
                }
                # Place randomly: inside a container or directly in a room
                # Weight towards room placement to avoid objects always
                # ending up in same containers across seeds
                placement = rng.choice(["container", "room", "room"])
                if placement == "container":
                    # Exclude the goal location container from hiding spots
                    all_containers = [
                        nid for nid, n in self.nodes.items()
                        if "CONTAINERS" in n.get("properties", [])
                        and nid != self.goal_location_id
                        and n["id"] != self.goal_location_id
                    ]
                    target = rng.choice(all_containers)
                    rel = "INSIDE"
                else:
                    # Place directly in a random room
                    room_name = rng.choice(self.ROOMS)
                    target = self.room_ids[room_name]
                    rel = "INSIDE"

                self.edges.append({
                    "from_id": oid, "to_id": target,
                    "relation_type": rel
                })
                placed_objects[obj_class].append(oid)

        # ── Noise objects (for Noisy-C-WAH) ──────────────────────────────
        noise_classes = ["book", "toy", "pen", "remote", "sock",
                         "towel", "pillow", "newspaper", "bottle", "cup"]
        for i in range(self.add_noise_objects):
            nid = _new_id()
            nc = noise_classes[i % len(noise_classes)]
            room = rng.choice(self.ROOMS)
            self.nodes[nid] = {
                "id": nid,
                "class_name": nc,
                "category": "Props",
                "states": ["GRABBABLE"],
                "properties": ["GRABBABLE"],
            }
            self.edges.append({
                "from_id": nid,
                "to_id": self.room_ids[room],
                "relation_type": "INSIDE"
            })

        # ── Public attrs expected by CoELA-style runners ──────────────────
        self.room_info = [
            {"class_name": rn, "id": rid}
            for rn, rid in self.room_ids.items()
        ]
        # Match CoELA format exactly: {predicate: [count, True, count]}
        self.goal_spec = {
            aid: {k: [v[0], True, v[0]] for k, v in self.goal_spec_raw.items()}
            for aid in self.agent_ids
        }

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, actions: Dict[int, Optional[str]],
             messages: Dict[int, Optional[str]] = None
             ) -> Tuple[List[dict], bool, dict]:
        """
        Execute one step for both agents simultaneously.

        actions: {agent_id: action_string or None}
          action_string format matches CoELA:
            "[walktowards] <room_name> (room_id)"
            "[grab] <obj_name> (obj_id)"
            "[open] <container_name> (container_id)"
            "[putback] <obj> (id) <surface> (id)"
            "[putin] <obj> (id) <container> (id)"

        Returns: (observations, done, info)
        """
        if messages:
            self.messages = [
                messages.get(self.agent_ids[0]),
                messages.get(self.agent_ids[1]),
            ]
        else:
            self.messages = [None, None]

        info = {}
        for agent_id, action in actions.items():
            if action is None:
                continue
            result = self._execute_action(agent_id, action)
            info[agent_id] = result

        self.step_count += 1
        self.done = (self.step_count >= self.max_episode_length
                     or self._check_goal_complete())

        obs = [self._get_observation(aid) for aid in self.agent_ids]
        return obs, self.done, info

    def _execute_action(self, agent_id: int, action: str) -> str:
        """Parse and apply one primitive action to the world graph."""
        action = action.strip()

        # ── Walk towards (move agent to room or furniture) ───────────────
        if action.startswith("[walktowards]"):
            parts = action.split()
            try:
                target_id = int(parts[-1].strip("()"))
            except (ValueError, IndexError):
                return "parse_error"
            target_node = self.nodes.get(target_id)
            if target_node is None:
                return "target_not_found"
            # If target is a room, move agent inside
            if target_node.get("category") == "Rooms":
                self._move_agent_to_room(agent_id, target_id)
                return "moved"
            # If target is furniture/object, move agent to its room
            obj_room = self._get_room_of(target_id)
            if obj_room:
                self._move_agent_to_room(agent_id, obj_room)
                # Add a CLOSE edge so agent can interact with it
                self.edges = [e for e in self.edges
                              if not (e["from_id"] == agent_id
                                      and e["to_id"] == target_id
                                      and e["relation_type"] == "CLOSE")]
                self.edges.append({
                    "from_id": agent_id,
                    "to_id": target_id,
                    "relation_type": "CLOSE"
                })
                return "moved_to_object_room"
            return "ok"

        # ── Open container ────────────────────────────────────────────────
        elif action.startswith("[open]"):
            parts = action.split()
            try:
                target_id = int(parts[-1].strip("()"))
            except (ValueError, IndexError):
                return "parse_error"
            node = self.nodes.get(target_id)
            if node and "CLOSED" in node.get("states", []):
                node["states"].remove("CLOSED")
                node["states"].append("OPEN")
                return "opened"
            return "already_open_or_missing"

        # ── Grab object ───────────────────────────────────────────────────
        elif action.startswith("[grab]"):
            parts = action.split()
            try:
                obj_id = int(parts[-1].strip("()"))
            except (ValueError, IndexError):
                return "parse_error"
            held = self._get_held_objects(agent_id)
            if len(held) >= 2:
                return "hands_full"
            # Remove object from its current location edge
            self.edges = [
                e for e in self.edges
                if not (e["from_id"] == obj_id
                        and e["relation_type"] in ("INSIDE", "ON"))
            ]
            # Add holding edge
            self.edges.append({
                "from_id": agent_id,
                "to_id": obj_id,
                "relation_type": "HOLDS_RH" if len(held) == 0 else "HOLDS_LH"
            })
            return "grabbed"

        # ── Put back on surface ────────────────────────────────────────────
        elif action.startswith("[putback]"):
            parts = action.split()
            try:
                # Format: [putback] <obj_name> (obj_id) <surf_name> (surf_id)
                ids = [int(p.strip("()")) for p in parts if p.startswith("(") and p.endswith(")")]
                if len(ids) < 2:
                    return "parse_error"
                obj_id, surf_id = ids[0], ids[1]
            except (ValueError, IndexError):
                return "parse_error"
            return self._place_object(agent_id, obj_id, surf_id, "ON")

        # ── Put in container ───────────────────────────────────────────────
        elif action.startswith("[putin]"):
            parts = action.split()
            try:
                obj_id = int(parts[2].strip("()"))
                cont_id = int(parts[-1].strip("()"))
            except (ValueError, IndexError):
                return "parse_error"
            return self._place_object(agent_id, obj_id, cont_id, "INSIDE")

        return "unrecognized_action"

    def _place_object(self, agent_id: int, obj_id: int,
                      target_id: int, relation: str) -> str:
        """Remove holding edge, add placement edge."""
        # Remove holding edge
        self.edges = [
            e for e in self.edges
            if not (e["from_id"] == agent_id
                    and e["to_id"] == obj_id
                    and e["relation_type"] in ("HOLDS_RH", "HOLDS_LH"))
        ]
        self.edges.append({
            "from_id": obj_id,
            "to_id": target_id,
            "relation_type": relation
        })
        return "placed"

    def _move_agent_to_room(self, agent_id: int, room_id: int):
        """Update agent's INSIDE edge to new room."""
        self.edges = [
            e for e in self.edges
            if not (e["from_id"] == agent_id
                    and e["relation_type"] == "INSIDE")
        ]
        self.edges.append({
            "from_id": agent_id,
            "to_id": room_id,
            "relation_type": "INSIDE"
        })

    def _get_held_objects(self, agent_id: int) -> List[int]:
        return [
            e["to_id"] for e in self.edges
            if e["from_id"] == agent_id
            and e["relation_type"] in ("HOLDS_RH", "HOLDS_LH")
        ]

    def _get_room_of(self, node_id: int) -> Optional[int]:
        """Return room_id that contains node_id, searching transitively."""
        for e in self.edges:
            if e["from_id"] == node_id and e["relation_type"] == "INSIDE":
                target = self.nodes.get(e["to_id"])
                if target and target.get("category") == "Rooms":
                    return e["to_id"]
                # Try one more level (object inside container inside room)
                return self._get_room_of(e["to_id"])
        return None

    def _get_agent_room(self, agent_id: int) -> Optional[dict]:
        for e in self.edges:
            if (e["from_id"] == agent_id
                    and e["relation_type"] == "INSIDE"):
                return self.nodes.get(e["to_id"])
        return None

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_observation(self, agent_id: int) -> dict:
        """
        Build a partial-observation graph for agent_id.
        Agent sees all nodes in its current room only.
        Matches CoELA's symbolic observation format exactly.
        """
        agent_room = self._get_agent_room(agent_id)
        if agent_room is None:
            return {"nodes": list(self.nodes.values()),
                    "edges": self.edges[:],
                    "messages": self.messages[:]}

        current_room_id = agent_room["id"]

        # Collect visible node ids: room + anything INSIDE this room
        visible_ids: set = {current_room_id}
        # Always include goal location node so agents resolve its ID immediately
        if self.goal_location_id != -1:
            visible_ids.add(self.goal_location_id)
        # Add held objects
        for e in self.edges:
            if (e["from_id"] == agent_id
                    and e["relation_type"] in ("HOLDS_RH", "HOLDS_LH")):
                visible_ids.add(e["to_id"])
        # Add objects/containers directly in this room
        for e in self.edges:
            if (e["to_id"] == current_room_id
                    and e["relation_type"] == "INSIDE"):
                visible_ids.add(e["from_id"])
        # Add objects inside OPEN containers in this room
        open_containers_in_room = {
            e["from_id"] for e in self.edges
            if e["to_id"] == current_room_id
            and e["relation_type"] == "INSIDE"
            and "OPEN" in self.nodes.get(e["from_id"], {}).get("states", [])
        }
        for e in self.edges:
            if (e["to_id"] in open_containers_in_room
                    and e["relation_type"] == "INSIDE"):
                visible_ids.add(e["from_id"])
        # Add both agents always (can see self and agent if same room)
        for aid in self.agent_ids:
            room = self._get_agent_room(aid)
            if room and room["id"] == current_room_id:
                visible_ids.add(aid)
        # Add all room nodes (agent knows room names)
        for rid in self.room_ids.values():
            visible_ids.add(rid)

        # Add CLOSE edges for nearby objects
        close_edges = []
        for nid in visible_ids:
            n = self.nodes.get(nid)
            if n and n.get("category") not in ("Rooms",):
                close_edges.append({
                    "from_id": agent_id,
                    "to_id": nid,
                    "relation_type": "CLOSE"
                })

        filtered_nodes = [
            copy.deepcopy(self.nodes[nid])
            for nid in visible_ids if nid in self.nodes
        ]
        filtered_edges = [
            copy.deepcopy(e) for e in self.edges
            if e["from_id"] in visible_ids and e["to_id"] in visible_ids
        ] + close_edges

        return {
            "nodes":    filtered_nodes,
            "edges":    filtered_edges,
            "messages": self.messages[:]
        }

    # ── Goal check ────────────────────────────────────────────────────────────

    def _check_goal_complete(self) -> bool:
        """Check if all subgoals satisfied."""
        satisfied, unsatisfied = self.check_progress()
        return len(unsatisfied) == 0

    def check_progress(self) -> Tuple[List[dict], dict]:
        """
        Returns (satisfied_nodes, unsatisfied_dict).
        Matches CoELA's check_progress interface.
        """
        id2node = self.nodes
        satisfied = []
        unsatisfied = {}

        for predicate, (count, _, _) in self.goal_spec_raw.items():
            parts = predicate.split("_")
            relation = parts[0]           # "on" or "inside"
            obj_class = parts[1]
            # goal location is already self.goal_location_id
            cnt = count

            rel_map = {"on": "ON", "inside": "INSIDE"}
            rel_type = rel_map.get(relation, relation.upper())

            for e in self.edges:
                if cnt == 0:
                    break
                if (e["relation_type"] == rel_type
                        and e["to_id"] == self.goal_location_id):
                    from_node = id2node.get(e["from_id"])
                    if (from_node
                            and from_node["class_name"] == obj_class):
                        satisfied.append(from_node)
                        cnt -= 1

            if cnt > 0:
                unsatisfied[predicate] = cnt

        return satisfied, unsatisfied

    # ── Metrics helpers ───────────────────────────────────────────────────────

    def get_goal_spec(self, agent_id: int) -> dict:
        return self.goal_spec.get(agent_id, {})

    def get_full_graph(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": self.edges[:]}

    @property
    def goal_location_str(self) -> str:
        node = self.nodes.get(self.goal_location_id, {})
        return f"<{node.get('class_name', '?')}> ({self.goal_location_id})"
