# BLADE: Belief-Driven Cooperative Agents

Training-free cooperative embodied agents for partially observable household environments using probabilistic belief graphs, partner-aware planning, and value-of-information communication.

## Overview

BDCA is a multi-agent embodied AI framework designed for cooperative household task completion in partially observable environments such as C-WAH / VirtualHome.

Unlike prior approaches that rely on deterministic text memory, BDCA maintains a probabilistic belief graph over object locations and uses information-theoretic communication policies to reduce duplicate effort and improve coordination efficiency.

The system is built around:

- Probabilistic belief tracking
- Partner intent modeling
- Belief-aware planning
- VOI-based selective communication
- Lightweight LLM-assisted decision selection
- Real VirtualHome + mock environment support

---

## Features

### Probabilistic Belief Graph
Maintains uncertainty-aware distributions over object locations instead of deterministic memories.

### Partner Intent Modeling
Predicts teammate goals, target objects, and collision risks.

### VOI Communication
Messages are sent only when their expected information value exceeds communication cost.

### Belief-Aware Planning
Actions are scored using:
- goal progress
- information gain
- travel cost
- redundancy risk

### Mock Environment
Fast lightweight Python-only environment for debugging and ablations.

### Real Simulator Integration
Supports C-WAH / VirtualHome cooperative tasks.

---

## Repository Structure

```text
bdca/
├── agents/              # BDCA agent + partner model
├── belief/              # probabilistic belief graph
├── communication/       # VOI communication policy
├── planner/             # belief-aware planner
├── env/                 # mock + real environment wrappers
├── bdca_utils/          # runner + helpers
├── tests/               # unit tests
├── run.py               # main entry point
├── config.py            # central configuration
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <your_repo_url>
cd bdca
```

Create environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running BDCA

### Mock Environment

Run debug episodes without LLM:

```bash
python run.py --env mock --episodes 5 --no-llm --verbose
```

Run evaluation:

```bash
python run.py --env mock --episodes 20
```

---

### Real VirtualHome Environment

```bash
python run.py --env real --episodes 5
```

---

## Command Line Arguments

| Argument | Description |
|---|---|
| `--episodes N` | Number of episodes |
| `--task T` | Task ID |
| `--env {mock,real}` | Environment selection |
| `--no-llm` | Disable LLM usage |
| `--noise N` | Add noisy distractor objects |
| `--verbose` | Print detailed step logs |
| `--out PATH` | Output JSON path |

---

## Metrics

BDCA tracks:

- success rate
- simulation steps
- duplicate effort
- communication count
- communication efficiency
- false plans
- entropy reduction
- LLM calls

---

## External Dependencies

The real environment requires:

- VirtualHome
- Unity executable
- C-WAH benchmark environment

These are not bundled inside this repository.

---

## Testing

Run unit tests:

```bash
PYTHONPATH=. python -m pytest tests/test_all.py -v
```

---

## Research Direction

Current Phase-1 focuses on:
- stable infrastructure
- probabilistic belief tracking
- VOI communication
- cooperative embodied planning

Future work includes:
- counterfactual communication utility
- instance-level belief tracking
- TDW-MAT integration
- joint task allocation
- belief-conditioned plan validation

---

## Citation

If you use this repository, please cite:

```bibtex
@misc{bdca2026,
  title={BDCA: Belief-Driven Cooperative Agents},
  author={},
  year={2026}
}
```

---

## License

MIT License
