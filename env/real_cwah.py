import sys
import os

# Strategy: import CoELA modules first, then insert bdca path
# This avoids the utils package conflict

_VH_SIM  = '/home/inspire/Documents/project/virtualhome/simulation'
_CWAH    = '/home/inspire/Documents/project/CoELA/cwah'
_BDCA    = '/home/inspire/Documents/project/bdca'

# Step 1: Add CoELA paths to import CoELA modules
for p in [_VH_SIM, _CWAH]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Step 2: Import CoELA's UnityEnvironment NOW (before bdca utils shadows it)
from envs.unity_environment import UnityEnvironment

# Step 3: Remove CoELA cwah from path so its 'agents' doesn't shadow bdca's
# But keep VH simulation path
if _CWAH in sys.path:
    sys.path.remove(_CWAH)

# Step 4: Ensure bdca is first (for agents, utils etc)
if _BDCA in sys.path:
    sys.path.remove(_BDCA)
sys.path.insert(0, _BDCA)


class RealCWAHEnv:
    """
    Wrapper around CoELA's UnityEnvironment matching MockCWAHEnv interface.
    """

    def __init__(self, task_id=0,
                 executable_path='/home/inspire/Documents/project/executable/linux_exec.v2.3.0.x86_64',
                 port=8080, seed=42,
                 add_noise_objects=0):

        self.task_id           = task_id
        self.port              = port
        self.seed              = seed
        self.add_noise_objects = add_noise_objects
        self.executable_path   = executable_path

        self.max_episode_length    = 250
        self.step_count            = 0
        self.done                  = False
        self._env                  = None

        self.all_containers_name   = []
        self.all_goal_objects_name = []
        self.all_room_name         = []
        self.room_info             = []
        self.goal_spec             = {}
        self.goal_spec_raw         = {}
        self.task_name             = ""
        self.goal_location_id      = -1
        self.goal_location_class   = ""
        self.agent_ids             = [1, 2]
        self.messages              = [None, None]

    # Map our task IDs to CoELA task names
    TASK_MAP = {
        0: 'prepare_food',
        1: 'put_dishwasher',
        2: 'put_fridge',
        3: 'setup_table',
    }

    def _load_task_set(self):
        import pickle
        path = '/home/inspire/Documents/project/CoELA/cwah/dataset/test_env_set_help.pik'
        return pickle.load(open(path, 'rb'))

    def _get_task_index(self, task_id, seed=0):
        """Find index in dataset matching our task_id."""
        target_name = self.TASK_MAP.get(task_id, 'put_dishwasher')
        matching = [i for i, d in enumerate(self._task_set)
                    if d.get('task_name', '') == target_name]
        if not matching:
            return 0
        return matching[seed % len(matching)]

    def _init_env(self):
        self._task_set = self._load_task_set()
        self._env = UnityEnvironment(
            num_agents         = 2,
            max_episode_length = self.max_episode_length,
            port_id            = self.port,
            env_task_set       = self._task_set,
            observation_types  = ['partial', 'partial'],
            use_editor         = False,
            executable_args    = {
                'file_name':   self.executable_path,
                'x_display':   0,
                'no_graphics': True,
            },
            base_port          = self.port,
        )

    def reset(self, task_id=None):
        if task_id is not None:
            self.task_id = task_id

        if self._env is None:
            self._init_env()

        self.step_count = 0
        self.done       = False
        self.messages   = [None, None]

        # Map our task_id to real dataset index
        real_idx = self._get_task_index(self.task_id, self.seed)
        obs = self._env.reset(task_id=real_idx)
        while obs is None:
            obs = self._env.reset(task_id=real_idx)

        self.all_containers_name   = self._env.all_containers_name
        self.all_goal_objects_name = self._env.all_goal_objects_name
        self.all_room_name         = self._env.all_room_name
        self.room_info             = self._env.room_info
        self.task_name             = getattr(self._env, 'task_name',
                                             f'task_{self.task_id}')
        # CoELA uses 0-indexed agents, BDCAAgent expects 1-indexed
        # Also convert count format: int -> [count, True, count]
        raw_spec = self._env.goal_spec
        raw0 = raw_spec.get(0, raw_spec.get(1, {}))
        # Normalise to [count, True, count] format
        self.goal_spec_raw = {}
        for k, v in raw0.items():
            if isinstance(v, list):
                self.goal_spec_raw[k] = v
            elif isinstance(v, (int, float)):
                self.goal_spec_raw[k] = [int(v), True, int(v)]
            else:
                self.goal_spec_raw[k] = [1, True, 1]
        self.goal_spec = {
            1: self.goal_spec_raw,
            2: self.goal_spec_raw,
        }

        # Resolve goal location
        self.goal_location_id    = -1
        self.goal_location_class = ""
        if self.goal_spec_raw:
            first_pred = list(self.goal_spec_raw.keys())[0]
            # predicate format: "inside_cutleryfork_104" or "on_plate_268"
            parts = first_pred.split("_")
            # Last part might be numeric ID directly
            last = parts[-1]
            if last.isdigit():
                self.goal_location_id = int(last)
                # Find class name from full graph
                graph = self._env.get_graph() if hasattr(self._env, "get_graph") else {}
                for node in graph.get("nodes", []):
                    if node.get("id") == self.goal_location_id:
                        self.goal_location_class = node.get("class_name", "")
                        break
            else:
                self.goal_location_class = last
                graph = self._env.get_graph() if hasattr(self._env, "get_graph") else {}
                for node in graph.get("nodes", []):
                    if node.get("class_name") == self.goal_location_class:
                        self.goal_location_id = node["id"]
                        break

        for o in (obs or []):
            if isinstance(o, dict) and 'messages' not in o:
                o['messages'] = [None, None]

        return obs

    def step(self, actions: dict, messages: dict = None):
        if messages:
            self.messages = [messages.get(1), messages.get(2)]

        # UnityEnvironment uses 0-indexed agents
        # Convert actions to real env format
        # None = agent does nothing this step (convert_action skips None agents)
        def clean_action(a):
            if a is None:
                return None
            if isinstance(a, str) and a.startswith('[send_message]'):
                return None
            # Ensure action has (id) format that convert_action needs
            if isinstance(a, str) and '(' not in a:
                return None  # malformed action — skip
            return a
        action_dict = {0: clean_action(actions.get(1)),
                       1: clean_action(actions.get(2))}

        # UnityEnvironment.step returns (obs, reward, done, info)
        # but obs comes from get_observations() separately
        result = self._env.step(action_dict)

        # Extract done and per-agent failure flags from result
        failed_exec = {}
        if isinstance(result, tuple):
            if len(result) >= 3:
                done = result[2]
            else:
                done = result[-1]
            # CoELA stores per-agent EXECUTION_GENERAL failures in result[3]
            if len(result) >= 4 and isinstance(result[3], dict):
                failed_exec = result[3].get("failed_exec", {})
        else:
            done = False

        self.step_count += 1
        self.done = bool(done) or self.step_count >= self.max_episode_length

        # Get observations separately (CoELA pattern)
        obs = self._env.get_observations()

        if obs is None:
            obs = [{}, {}]

        for i, o in enumerate(obs):
            if isinstance(o, dict):
                o['messages'] = [
                    self.messages[1] if i == 0 else self.messages[0],
                    None
                ]

        return obs, self.done, {"failed_exec": failed_exec}

    def check_progress(self):
        if _CWAH not in sys.path:
            sys.path.append(_CWAH)
        try:
            from utils import utils_environment as env_utils
            graph = self._env.get_graph()
            # Use the env's own goal_spec (0-indexed, correct format)
            gs = self._env.goal_spec.get(0, {})
            satisfied, unsatisfied = env_utils.check_progress(graph, gs)
            # Convert to format runner expects:
            # satisfied: list of node dicts
            # unsatisfied: {predicate: remaining_count}
            sat_nodes = []
            for pred, items in satisfied.items():
                for item in items:
                    if item is not None and isinstance(item, str):
                        sat_nodes.append({"class_name": pred.split("_")[1],
                                          "id": item})
            return sat_nodes, unsatisfied
        except Exception as e:
            return [], {}
        finally:
            if _CWAH in sys.path:
                sys.path.remove(_CWAH)

    def close(self):
        if self._env:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def __del__(self):
        self.close()
