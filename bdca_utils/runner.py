"""
BDCA Episode Runner
===================
Runs full episodes and collects all metrics including the new
ones that CoELA and REVECA don't report:
  - duplicate_effort_rate
  - false_plan_rate
  - belief_entropy_reduction
  - communication_efficiency
  - belief_accuracy (when ground truth available)
"""
from __future__ import annotations
import json, time, copy
from typing import Dict, List, Optional, Any

from env.real_cwah import RealCWAHEnv
from agents.bdca_agent import BDCAAgent
from config import MAX_STEPS


class EpisodeRunner:

    def __init__(self, env: RealCWAHEnv,
                 agent1: BDCAAgent,
                 agent2: BDCAAgent,
                 max_steps: int = MAX_STEPS,
                 verbose: bool = False):
        self.env       = env
        self.agents    = [agent1, agent2]
        self.max_steps = max_steps
        self.verbose   = verbose

    def run_episode(self, task_id: int = 0,
                    seed: int = 42) -> dict:
        """Run one complete episode. Returns metrics dict."""
        self.env.task_id = task_id
        self.env.seed    = task_id * 100 + seed
        self.env.rng     = __import__("random").Random(self.env.seed)

        obs_list = self.env.reset(task_id=task_id)

        for i, agent in enumerate(self.agents):
            obs = obs_list[i]
            agent.reset(
                obs,
                self.env.all_containers_name,
                self.env.all_goal_objects_name,
                self.env.all_room_name,
                self.env.room_info,
                self.env.goal_spec[agent.agent_id],
            )

        step        = 0
        done        = False
        total_steps = 0
        messages_out= {a.agent_id: None for a in self.agents}
        success     = False

        t_start = time.time()

        while not done and step < self.max_steps:
            actions  = {}
            messages = {}

            for i, agent in enumerate(self.agents):
                obs = obs_list[i]
                # Inject latest messages into observation
                msg_list = [None, None]
                for j, other in enumerate(self.agents):
                    if other.agent_id != agent.agent_id:
                        msg_list[j] = messages_out.get(other.agent_id)
                obs["messages"] = msg_list

                action, info = agent.get_action(
                    obs, self.env.goal_spec[agent.agent_id])

                # Separate message actions
                if action and action.startswith("[send_message]"):
                    content = action.split("<")[-1].rstrip(">")
                    messages[agent.agent_id] = content
                    actions[agent.agent_id]  = None  # still count step
                else:
                    actions[agent.agent_id]   = action
                    messages[agent.agent_id]  = None

                if self.verbose:
                    print(f"  Step {step:3d} | {agent.agent_name} | "
                          f"plan={agent.plan} | action={action}")

            messages_out = messages

            try:
                obs_list, done, step_info = self.env.step(actions, messages)
            except (IndexError, KeyError, TypeError) as e:
                # Real env action parse error — skip this step
                print(f"Step error (skipped): {e}")
                step += 1
                continue
            step      += 1
            total_steps = step

            # ── Detect failed actions — clear plan on EXECUTION_GENERAL ──────
            # Prevents loops like repeatedly sending [open] on an already-open
            # container when the partial obs hasn't caught up yet.
            try:
                failed_exec = (step_info.get("failed_exec", {})
                               if isinstance(step_info, dict) else {})
                if failed_exec:
                    for agent in self.agents:
                        # CoELA uses 0-indexed agents; ours are 1-indexed
                        agent_failed = (failed_exec.get(agent.agent_id - 1)
                                        or failed_exec.get(agent.agent_id))
                        if agent_failed:
                            # Remove the failed plan from LLM context so it
                            # doesn't reinforce the impossible action
                            if agent.action_history:
                                agent.action_history.pop()
                            agent.plan = None  # force re-plan next step
            except Exception:
                pass

            # ── Sync ground truth satisfaction to agents every step ──────────
            # Critical: prevents agents looping on already-completed subgoals
            try:
                _, unsatisfied_gt = self.env.check_progress()
                for agent in self.agents:
                    for pred, remaining in unsatisfied_gt.items():
                        if pred in agent.unsatisfied:
                            agent.unsatisfied[pred] = max(0, int(remaining))
            except Exception:
                pass

            if done:
                _, unsatisfied = self.env.check_progress()
                success = (len(unsatisfied) == 0 or
                           all(v <= 0 for v in unsatisfied.values()))
                break

        elapsed = time.time() - t_start

        # ── Collect metrics ───────────────────────────────────────────────────
        _, unsatisfied_final = self.env.check_progress()
        # Success = all unsatisfied counts are 0 or negative
        final_success = all(v <= 0 for v in unsatisfied_final.values())
        if not success:
            success = final_success
        n_subgoals_total     = sum(
            v[0] for v in self.env.goal_spec_raw.values())
        n_satisfied          = n_subgoals_total - sum(
            unsatisfied_final.values())

        # Per-agent new metrics
        dup_effort  = sum(a.metrics["duplicate_effort_attempts"]
                          for a in self.agents)
        false_plans = sum(a.metrics["false_plan_count"]
                          for a in self.agents)
        comm_sent   = sum(a.metrics["comm_sent"] for a in self.agents)
        comm_voi    = sum(a.metrics["comm_voi_total"] for a in self.agents)
        comm_eff    = (comm_voi / max(comm_sent, 1))

        # Belief entropy reduction
        all_entropy_logs = []
        for a in self.agents:
            el = a.metrics.get("belief_entropy_log", [])
            if el:
                all_entropy_logs.extend(el)

        initial_entropy = all_entropy_logs[0]  if all_entropy_logs else 0
        final_entropy   = all_entropy_logs[-1] if all_entropy_logs else 0
        entropy_reduction = max(0, initial_entropy - final_entropy)

        return {
            # ── Standard metrics (match CoELA/REVECA) ──────────────────────
            "task_id":          task_id,
            "task_name":        self.env.task_name,
            "success":          success,
            "simulation_steps": total_steps,
            "subgoals_satisfied": n_satisfied,
            "subgoals_total":     n_subgoals_total,

            # ── New BDCA metrics ────────────────────────────────────────────
            "duplicate_effort_attempts": dup_effort,
            "false_plan_count":          false_plans,
            "comm_sent":                 comm_sent,
            "communication_efficiency":  round(comm_eff, 3),
            "belief_entropy_reduction":  round(entropy_reduction, 3),
            "initial_entropy":           round(initial_entropy, 3),
            "final_entropy":             round(final_entropy, 3),

            # ── Timing ─────────────────────────────────────────────────────
            "elapsed_seconds": round(elapsed, 1),
            "llm_calls": sum(a._llm_calls for a in self.agents),
        }


def run_experiments(n_episodes: int = 10,
                    task_ids: Optional[List[int]] = None,
                    noise_objects: int = 0,
                    verbose: bool = False,
                    api_key: str = "") -> dict:
    """
    Run N episodes across all task types and aggregate results.

    This is the Phase 1 evaluation loop.
    Run without an API key for logic testing (LLM falls back to greedy).
    Run with ANTHROPIC_API_KEY set for full LLM-driven runs.
    """
    import argparse
    args = argparse.Namespace(anthropic_api_key=api_key)

    if task_ids is None:
        task_ids = list(range(4)) * max(1, n_episodes // 4)
        task_ids = task_ids[:n_episodes]

    results = []

    for ep_idx, task_id in enumerate(task_ids):
        env = RealCWAHEnv(task_id=task_id, seed=ep_idx * 7,
                          add_noise_objects=noise_objects)

        agent1 = BDCAAgent(agent_id=1, char_index=0, args=args)
        agent2 = BDCAAgent(agent_id=2, char_index=1, args=args)

        runner = EpisodeRunner(env, agent1, agent2, verbose=verbose)
        result = runner.run_episode(task_id=task_id, seed=ep_idx)
        results.append(result)

        if verbose or True:
            print(f"Episode {ep_idx+1:2d}/{len(task_ids)} | "
                  f"Task: {result['task_name'][:20]:<20} | "
                  f"Steps: {result['simulation_steps']:3d} | "
                  f"Success: {result['success']} | "
                  f"DupEff: {result['duplicate_effort_attempts']:2d} | "
                  f"FalsePlan: {result['false_plan_count']:2d} | "
                  f"Comm: {result['comm_sent']:2d} | "
                  f"Entropy↓: {result['belief_entropy_reduction']:.2f}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(results)
    agg = {
        "n_episodes":              n,
        "success_rate":            round(sum(r["success"] for r in results) / n, 3),
        "avg_simulation_steps":    round(sum(r["simulation_steps"] for r in results) / n, 1),
        "avg_subgoals_satisfied":  round(sum(r["subgoals_satisfied"] for r in results) / n, 2),
        "avg_duplicate_effort":    round(sum(r["duplicate_effort_attempts"] for r in results) / n, 2),
        "avg_false_plans":         round(sum(r["false_plan_count"] for r in results) / n, 2),
        "avg_comm_sent":           round(sum(r["comm_sent"] for r in results) / n, 2),
        "avg_comm_efficiency":     round(sum(r["communication_efficiency"] for r in results) / n, 3),
        "avg_entropy_reduction":   round(sum(r["belief_entropy_reduction"] for r in results) / n, 3),
        "total_llm_calls":         sum(r["llm_calls"] for r in results),
        "episodes":                results,
    }
    return agg
