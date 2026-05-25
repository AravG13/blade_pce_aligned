"""
BDCA Phase 1 Test Suite
=======================
Tests every module independently + an integration test.
Run with:  python tests/test_all.py
All tests must pass before proceeding to Phase 2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
import math


# ─────────────────────────────────────────────────────────────────────────────
class TestBeliefGraph(unittest.TestCase):

    def _make_bg(self):
        from belief.belief_graph import BeliefGraph
        bg = BeliefGraph("alice", ["cupcake", "apple", "juice"], "bob")
        bg.initialize_priors(
            rooms=["kitchen", "livingroom", "bedroom", "bathroom"],
            containers=["fridge", "cabinet"],
            goal_location="coffeetable (268)",
        )
        return bg

    def test_initialization_creates_priors(self):
        bg = self._make_bg()
        conf = bg.get_confidence("cupcake", "location")
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_observation_update_raises_confidence(self):
        bg = self._make_bg()
        obs = {
            "current_room": "kitchen",
            "visible_objects": [{"id": 216, "class_name": "cupcake"}],
            "visible_containers": [],
            "inspected_containers": [],
            "held_objects": [],
            "teammate_visible": False,
            "completed_subgoals": [],
        }
        initial_conf = bg.get_confidence("cupcake", "location")
        bg.update_from_observation(obs, step=5)
        new_conf = bg.get_confidence("cupcake", "location", "kitchen")
        self.assertGreater(new_conf, initial_conf)
        self.assertAlmostEqual(new_conf, 0.95, places=2)

    def test_mutual_exclusivity(self):
        bg = self._make_bg()
        obs = {
            "current_room": "kitchen",
            "visible_objects": [{"id": "cupcake_216", "class_name": "cupcake"}],
            "visible_containers": [], "inspected_containers": [],
            "held_objects": [], "teammate_visible": False,
            "completed_subgoals": [],
        }
        bg.update_from_observation(obs, step=1)
        alts = bg.get_all_location_hypotheses("cupcake")
        total = sum(alts.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_temporal_decay_reduces_confidence(self):
        bg = self._make_bg()
        obs = {
            "current_room": "kitchen",
            "visible_objects": [{"id": "cupcake_216", "class_name": "cupcake"}],
            "visible_containers": [], "inspected_containers": [],
            "held_objects": [], "teammate_visible": False,
            "completed_subgoals": [],
        }
        bg.update_from_observation(obs, step=1)
        conf_before = bg.get_confidence("cupcake", "location")
        bg.apply_decay(current_step=50, teammate_last_room="kitchen")
        conf_after = bg.get_confidence("cupcake", "location")
        self.assertLess(conf_after, conf_before)

    def test_message_update_subgoal_complete(self):
        bg = self._make_bg()
        msg = {"type": "subgoal_complete", "object_id": "apple",
               "sender": "bob"}
        bg.update_from_message(msg, step=10)
        self.assertEqual(bg.get_subgoal_status("apple"), "complete")

    def test_message_update_object_found(self):
        bg = self._make_bg()
        msg = {"type": "object_found", "object_id": "juice",
               "room": "bedroom", "sender": "bob"}
        bg.update_from_message(msg, step=10)
        loc, conf = bg.best_hypothesis("juice", "location")
        self.assertEqual(loc, "bedroom")
        self.assertAlmostEqual(conf, 0.80, places=2)

    def test_entropy_decreases_after_observation(self):
        bg = self._make_bg()
        entropy_before = bg.total_entropy()
        obs = {
            "current_room": "kitchen",
            "visible_objects": [
                {"id": 1, "class_name": "cupcake"},
                {"id": 2, "class_name": "apple"},
            ],
            "visible_containers": [], "inspected_containers": [],
            "held_objects": [], "teammate_visible": False,
            "completed_subgoals": [],
        }
        bg.update_from_observation(obs, step=1)
        entropy_after = bg.total_entropy()
        self.assertLess(entropy_after, entropy_before)

    def test_uninspected_containers(self):
        bg = self._make_bg()
        uninspected = bg.uncertain_objects(threshold=1.0)
        # All objects uncertain before any observations
        self.assertEqual(len(uninspected), 3)

    def test_summary_is_string(self):
        bg = self._make_bg()
        s = bg.summary()
        self.assertIsInstance(s, str)
        self.assertIn("cupcake", s)
        self.assertIn("entropy", s)

    def test_redistribute_when_container_empty(self):
        bg = self._make_bg()
        from belief.belief_graph import BeliefFact, SRC_OBS, F_LOCATION
        # Force cupcake to be believed inside fridge
        bg._update_location("cupcake", "fridge", 1, SRC_OBS, 0.8)
        obs = {
            "current_room": "kitchen",
            "visible_objects": [],
            "visible_containers": [],
            "inspected_containers": [{"id": "fridge", "contents": []}],
            "held_objects": [], "teammate_visible": False,
            "completed_subgoals": [],
        }
        bg.update_from_observation(obs, step=2)
        fridge_conf = bg.get_confidence("cupcake", "location", "fridge")
        # Should have redistributed away from fridge
        self.assertLess(fridge_conf, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
class TestPartnerModel(unittest.TestCase):

    def _make(self):
        from agents.partner_model import PartnerIntentModel
        return PartnerIntentModel(
            agent_id="alice", teammate_id="bob",
            goal_objects=["cupcake", "apple", "juice"],
            llm_fn=None,
        )

    def _make_bg(self):
        from belief.belief_graph import BeliefGraph
        bg = BeliefGraph("alice", ["cupcake", "apple", "juice"], "bob")
        bg.initialize_priors(
            ["kitchen", "livingroom", "bedroom", "bathroom"],
            ["fridge", "cabinet"], "coffeetable (268)")
        return bg

    def test_prediction_returns_object(self):
        from agents.partner_model import PartnerPrediction
        pm  = self._make()
        bg  = self._make_bg()
        pred = pm.predict(bg, [], 5, ["cupcake", "apple"])
        self.assertIsInstance(pred, PartnerPrediction)

    def test_prediction_has_collision_risk(self):
        pm   = self._make()
        bg   = self._make_bg()
        pred = pm.predict(bg, [], 5, ["cupcake", "apple"])
        self.assertIn("cupcake", pred.collision_risk)
        self.assertIn("apple",   pred.collision_risk)

    def test_cache_freshness(self):
        pm   = self._make()
        bg   = self._make_bg()
        pred1 = pm.predict(bg, [], 5, ["cupcake"])
        pred2 = pm.predict(bg, [], 6, ["cupcake"])  # within cache window
        self.assertIs(pred1, pred2)                  # same object

    def test_cache_expires(self):
        pm   = self._make()
        bg   = self._make_bg()
        pred1 = pm.predict(bg, [], 1,  ["cupcake"])
        pred2 = pm.predict(bg, [], 10, ["cupcake"])  # beyond cache
        self.assertIsNot(pred1, pred2)

    def test_message_boosts_room_confidence(self):
        pm = self._make()
        bg = self._make_bg()
        msgs = [{"content": "I found apple in kitchen"}]
        pred = pm.predict(bg, msgs, 1, ["apple"])
        # Rule should pick up "kitchen" from message
        self.assertEqual(pred.likely_room, "kitchen")


# ─────────────────────────────────────────────────────────────────────────────
class TestVOIPolicy(unittest.TestCase):

    def _make(self):
        from communication.voi_policy import VOICommunicationPolicy
        return VOICommunicationPolicy(
            agent_id="alice", teammate_id="bob",
            goal_objects=["cupcake", "apple", "juice"],
            llm_fn=None,
        )

    def _make_bg(self):
        from belief.belief_graph import BeliefGraph
        bg = BeliefGraph("alice", ["cupcake", "apple", "juice"], "bob")
        bg.initialize_priors(
            ["kitchen","livingroom","bedroom","bathroom"],
            ["fridge","cabinet"], "coffeetable (268)")
        return bg

    def _make_pred(self, room="kitchen", risk=0.7):
        from agents.partner_model import PartnerPrediction
        return PartnerPrediction(
            likely_room="kitchen", room_conf=0.8,
            likely_held=[], likely_target="cupcake",
            target_conf=0.75,
            collision_risk={"cupcake": risk, "apple": 0.1, "juice": 0.05},
            predicted_at=0,
        )

    def test_subgoal_complete_always_sends(self):
        policy = self._make()
        bg     = self._make_bg()
        pred   = self._make_pred()
        dec    = policy.decide(bg, pred, 10, ["apple"],
                               just_completed="cupcake", my_room="kitchen")
        self.assertTrue(dec.should_send)
        self.assertEqual(dec.message_type, "subgoal_complete")

    def test_high_collision_risk_triggers_query(self):
        policy = self._make()
        bg     = self._make_bg()
        # Put cupcake in kitchen with high confidence
        from belief.belief_graph import SRC_OBS
        bg._update_location("cupcake", "kitchen", 1, SRC_OBS, 0.90)
        pred   = self._make_pred(risk=0.75)
        dec    = policy.decide(bg, pred, 10, ["cupcake"],
                               my_room="bedroom")
        # High risk should produce positive VOI → send query
        self.assertGreater(dec.voi, 0)

    def test_cooldown_prevents_spam(self):
        policy = self._make()
        bg     = self._make_bg()
        pred   = self._make_pred(risk=0.9)
        from belief.belief_graph import SRC_OBS
        bg._update_location("cupcake", "kitchen", 1, SRC_OBS, 0.90)
        dec1 = policy.decide(bg, pred, 10, ["cupcake"], my_room="bedroom")
        dec2 = policy.decide(bg, pred, 11, ["cupcake"], my_room="bedroom")
        # Second call within cooldown should not send
        if dec1.should_send:
            self.assertFalse(dec2.should_send)

    def test_low_risk_no_query(self):
        policy = self._make()
        bg     = self._make_bg()
        pred   = self._make_pred(risk=0.05)
        dec    = policy.decide(bg, pred, 10, ["cupcake"], my_room="bedroom")
        self.assertFalse(dec.should_send)


# ─────────────────────────────────────────────────────────────────────────────
class TestMockEnvironment(unittest.TestCase):

    def test_reset_returns_two_observations(self):
        from env.mock_cwah import MockCWAHEnv
        env = MockCWAHEnv(task_id=0)
        obs = env.reset()
        self.assertEqual(len(obs), 2)
        for o in obs:
            self.assertIn("nodes", o)
            self.assertIn("edges", o)

    def test_step_decrements_done_at_max(self):
        from env.mock_cwah import MockCWAHEnv
        env = MockCWAHEnv(task_id=0)
        env.reset()
        env.max_episode_length = 3
        for _ in range(3):
            obs, done, info = env.step({1: None, 2: None})
        self.assertTrue(done)

    def test_goal_location_exists_in_nodes(self):
        from env.mock_cwah import MockCWAHEnv
        env = MockCWAHEnv(task_id=0)
        env.reset()
        ids = [n["id"] for n in env.nodes.values()]
        self.assertIn(env.goal_location_id, ids)

    def test_walk_action_moves_agent(self):
        from env.mock_cwah import MockCWAHEnv
        env = MockCWAHEnv(task_id=0, seed=1)
        env.reset()
        agent_id = env.agent_ids[0]
        room_before = env._get_agent_room(agent_id)["class_name"]

        # Find a different room to walk to
        for rname, rid in env.room_ids.items():
            if rname != room_before:
                env.step({agent_id: f"[walktowards] <{rname}> ({rid})",
                           env.agent_ids[1]: None})
                break

        room_after = env._get_agent_room(agent_id)["class_name"]
        self.assertNotEqual(room_before, room_after)

    def test_noisy_objects_added(self):
        from env.mock_cwah import MockCWAHEnv
        env_clean = MockCWAHEnv(task_id=0, add_noise_objects=0)
        env_noisy = MockCWAHEnv(task_id=0, add_noise_objects=10)
        env_clean.reset()
        env_noisy.reset()
        self.assertGreater(len(env_noisy.nodes), len(env_clean.nodes))

    def test_check_progress_initially_empty(self):
        from env.mock_cwah import MockCWAHEnv
        env = MockCWAHEnv(task_id=1)  # Wash dishes — 2 subgoals
        env.reset()
        sat, unsat = env.check_progress()
        self.assertEqual(len(sat), 0)
        self.assertGreater(len(unsat), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestBDCAAgent(unittest.TestCase):

    def _make_agent_and_env(self, task_id=0):
        import argparse
        from env.mock_cwah  import MockCWAHEnv
        from agents.bdca_agent import BDCAAgent
        env    = MockCWAHEnv(task_id=task_id, seed=42)
        obs    = env.reset()
        args   = argparse.Namespace(anthropic_api_key="")
        agent  = BDCAAgent(agent_id=1, char_index=0, args=args)
        agent.reset(
            obs[0],
            env.all_containers_name,
            env.all_goal_objects_name,
            env.all_room_name,
            env.room_info,
            env.goal_spec[1],
        )
        return agent, env, obs

    def test_reset_initialises_belief(self):
        agent, env, obs = self._make_agent_and_env()
        self.assertIsNotNone(agent.belief)
        self.assertGreater(len(agent.belief.goal_objects), 0)

    def test_reset_initialises_planner(self):
        agent, env, obs = self._make_agent_and_env()
        self.assertIsNotNone(agent.planner)

    def test_get_action_returns_string_or_none(self):
        agent, env, obs = self._make_agent_and_env()
        action, info = agent.get_action(obs[0], env.goal_spec[1])
        self.assertIsInstance(info, dict)
        # action can be None (wait) or a string
        if action is not None:
            self.assertIsInstance(action, str)

    def test_full_step_doesnt_crash(self):
        import argparse
        from env.mock_cwah  import MockCWAHEnv
        from agents.bdca_agent import BDCAAgent
        env  = MockCWAHEnv(task_id=1, seed=10)
        obs  = env.reset()
        args = argparse.Namespace(anthropic_api_key="")
        agents = [BDCAAgent(agent_id=i+1, char_index=i, args=args)
                  for i in range(2)]
        for i, ag in enumerate(agents):
            ag.reset(obs[i], env.all_containers_name,
                     env.all_goal_objects_name, env.all_room_name,
                     env.room_info, env.goal_spec[ag.agent_id])
        for step in range(5):
            actions = {}
            for i, ag in enumerate(agents):
                obs[i]["messages"] = [None, None]
                a, _ = ag.get_action(obs[i], env.goal_spec[ag.agent_id])
                actions[ag.agent_id] = a
            obs, done, _ = env.step(actions)
            if done:
                break

    def test_metrics_tracked(self):
        agent, env, obs = self._make_agent_and_env()
        for _ in range(3):
            obs[0]["messages"] = [None, None]
            action, info = agent.get_action(obs[0], env.goal_spec[1])
            obs, done, _ = env.step({1: action, 2: None})
            if done:
                break
        self.assertIn("duplicate_effort_attempts", agent.metrics)
        self.assertIn("belief_entropy_log", agent.metrics)
        self.assertGreater(len(agent.metrics["belief_entropy_log"]), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration(unittest.TestCase):

    def test_10_step_episode_no_crash(self):
        """Full integration: env + 2 BDCA agents for 10 steps."""
        import argparse
        from env.mock_cwah  import MockCWAHEnv
        from agents.bdca_agent import BDCAAgent
        from bdca_utils.runner import EpisodeRunner

        env  = MockCWAHEnv(task_id=0, seed=99)
        obs  = env.reset()
        args = argparse.Namespace(anthropic_api_key="")
        ag1  = BDCAAgent(1, 0, args)
        ag2  = BDCAAgent(2, 1, args)
        for i, ag in enumerate([ag1, ag2]):
            ag.reset(obs[i], env.all_containers_name,
                     env.all_goal_objects_name, env.all_room_name,
                     env.room_info, env.goal_spec[ag.agent_id])

        runner = EpisodeRunner(env, ag1, ag2, max_steps=10)
        result = runner.run_episode(task_id=0, seed=0)

        self.assertIn("simulation_steps", result)
        self.assertIn("duplicate_effort_attempts", result)
        self.assertIn("belief_entropy_reduction", result)
        self.assertLessEqual(result["simulation_steps"], 10)

    def test_result_dict_has_all_keys(self):
        import argparse
        from env.mock_cwah  import MockCWAHEnv
        from agents.bdca_agent import BDCAAgent
        from bdca_utils.runner import EpisodeRunner

        env  = MockCWAHEnv(task_id=1, seed=5)
        obs  = env.reset()
        args = argparse.Namespace(anthropic_api_key="")
        ag1  = BDCAAgent(1, 0, args)
        ag2  = BDCAAgent(2, 1, args)
        for i, ag in enumerate([ag1, ag2]):
            ag.reset(obs[i], env.all_containers_name,
                     env.all_goal_objects_name, env.all_room_name,
                     env.room_info, env.goal_spec[ag.agent_id])

        runner = EpisodeRunner(env, ag1, ag2, max_steps=5)
        result = runner.run_episode(task_id=1)

        required_keys = [
            "success", "simulation_steps", "subgoals_satisfied",
            "duplicate_effort_attempts", "false_plan_count",
            "comm_sent", "communication_efficiency",
            "belief_entropy_reduction",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestBeliefGraph, TestPartnerModel, TestVOIPolicy,
                TestMockEnvironment, TestBDCAAgent, TestIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
