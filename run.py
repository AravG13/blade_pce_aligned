import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#!/usr/bin/env python3
"""
BDCA Phase 1 — Main Entry Point
================================
Usage:
    # Logic test (no API key required):
    python run.py --episodes 10 --no-llm

    # Full LLM run:
    OPENAI_API_KEY=sk-ant-... python run.py --episodes 10

    # Noisy environment (like Noisy-C-WAH):
    python run.py --episodes 10 --noise 20 --no-llm

    # Verbose single episode:
    python run.py --episodes 1 --verbose --no-llm
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Run BDCA Phase 1")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--noise",    type=int, default=0,
                        help="Number of dummy objects (Noisy-C-WAH mode)")
    parser.add_argument("--verbose",  action="store_true",
                        help="Print step-by-step actions")
    parser.add_argument("--no-llm",   action="store_true",
                        help="Disable LLM calls (greedy fallback)")
    parser.add_argument("--out",      type=str, default="results/phase1.json",
                        help="Output path for results JSON")
    parser.add_argument("--task",     type=int, default=-1,
                        help="Run a specific task id (0-3), -1 = all")
    args = parser.parse_args()

    api_key = "" if args.no_llm else os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.no_llm:
        print("Warning: OPENAI_API_KEY not set. Using greedy fallback.")
        print("         Pass --no-llm to suppress this warning.")

    task_ids = (
        [args.task] * args.episodes
        if args.task >= 0
        else [i % 4 for i in range(args.episodes)]
    )

    print(f"\n{'='*60}")
    print(f"  BDCA Phase 1 Experiment")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Noise objs: {args.noise}")
    print(f"  LLM:        {'OFF (greedy)' if not api_key else 'ON'}")
    print(f"{'='*60}\n")

    from bdca_utils.runner import run_experiments
    results = run_experiments(
        n_episodes    = args.episodes,
        task_ids      = task_ids,
        noise_objects = args.noise,
        verbose       = args.verbose,
        api_key       = api_key,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY ({results['n_episodes']} episodes)")
    print(f"{'='*60}")
    print(f"  Success rate:          {results['success_rate']:.1%}")
    print(f"  Avg simulation steps:  {results['avg_simulation_steps']:.1f}")
    print(f"  Avg subgoals done:     {results['avg_subgoals_satisfied']:.2f}")
    print(f"  ── New BDCA Metrics ──────────────────────────────")
    print(f"  Avg duplicate effort:  {results['avg_duplicate_effort']:.2f}")
    print(f"  Avg false plans:       {results['avg_false_plans']:.2f}")
    print(f"  Avg comm sent:         {results['avg_comm_sent']:.2f}")
    print(f"  Comm efficiency (VOI): {results['avg_comm_efficiency']:.3f}")
    print(f"  Entropy reduction:     {results['avg_entropy_reduction']:.3f}")
    print(f"  Total LLM calls:       {results['total_llm_calls']}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {args.out}")


if __name__ == "__main__":
    main()
