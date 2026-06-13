# Autonomous Drone Rescue — Dynamic Programming (Value Iteration)

> A 6×6 grid-world MDP simulation of an autonomous rescue drone, solved with Value Iteration (Dynamic Programming). The drone must rescue 3 civilians while managing battery, avoiding danger zones and obstacles, and handling stochastic wind disturbances — all under the Bellman Optimality Equation.

---

## Outputs

| Policy Map | Value Heatmap |
|---|---|
| ![Policy Map](outputs/policy_map.png) | ![Value Heatmap](outputs/value_heatmap.png) |

| Trajectory | Convergence |
|---|---|
| ![Trajectory](outputs/trajectory.png) | ![Convergence](outputs/convergence.png) |

---

## Features

- **Custom 6×6 Grid-World MDP** — purpose-built environment (Group ID 176 configuration) with:
  - 3 rescue targets (R)
  - 2 charging stations (C)
  - 4 danger zones (D)
  - 3 blocked cells (X)
  - 2 wind zones (W)
- **State Space** — `state = (row, col, battery, rescued_targets)` → 33 valid cells × 10 battery levels × 8 rescue combinations = **2,640 states**
- **5 Actions** — UP, DOWN, LEFT, RIGHT, HOVER
- **Stochastic Wind Dynamics** — 30% chance a directional action deviates to a random cardinal direction on wind cells
- **Battery Management** — every action costs 1 unit; entering a charging station refills to max (+5 reward); hovering on a charger adds +2 battery (no reward)
- **Value Iteration Solver** — implements the Bellman Optimality Equation with convergence threshold θ = 10⁻³ and discount factor γ = 0.9
- **Visualisations** — policy map, value-function heatmap, greedy trajectory simulation, and convergence curve, all saved to `outputs/`

---

## Project Structure

```
Drone_Rescue/
├── src/
│   ├── environment.py      # DroneEnv — MDP definition (Group 176 config, transitions, rewards)
│   ├── value_iteration.py  # Bellman optimality solver with convergence tracking
│   ├── visualization.py    # Policy map, value heatmap, trajectory & convergence plots
│   └── main.py             # Entry point — runs the full pipeline
├── outputs/
│   ├── policy_map.png
│   ├── value_heatmap.png
│   ├── trajectory.png
│   └── convergence.png
├── docs/
│   └── report.md           # Full academic report
└── requirements.txt
```

---

## Architecture / Pipeline

```
DroneEnv (environment.py)
  │  • Defines grid, rewards, transitions, wind stochasticity
  │  • enumerate_states() → 2,640 states
  │
  ▼
value_iteration() (value_iteration.py)
  │  • Applies Bellman Optimality Equation:
  │      V*(s) = max_a Σ_s' P(s'|s,a) [R(s,a,s') + γ·V*(s')]
  │  • Iterates until max|ΔV| < θ (1e-3)
  │  • Returns V*, π*, convergence info
  │
  ▼
visualization.py
  │  • plot_policy()           → policy_map.png
  │  • plot_value_heatmap()     → value_heatmap.png
  │  • simulate_trajectory()    → greedy rollout using π*
  │  • plot_trajectory()        → trajectory.png
  │  • plot_convergence()       → convergence.png
  │
  ▼
main.py — orchestrates the full pipeline and prints diagnostics
```

---

## Environment Configuration (Group ID 176)

| Parameter | Value | Rule |
|---|---|---|
| Grid size | 6×6 | Last digit 5–9 → 6×6 grid |
| Rescue targets | 3 | Per spec for 6×6 |
| Charging stations | 2 | Per spec for 6×6 |
| Danger zones | 4 | Per spec for 6×6 |
| Blocked cells | 3 | Per spec for 6×6 |
| Max battery | 10 | Even last digit (6) |
| Wind probability | 30% | Last digit in 5–9 (6) |
| Max episode steps | 75 | 6×6 grid step limit |

### Grid Layout

```
     Col:  0    1    2    3    4    5
Row 0:     S    F    W    D    R    F
Row 1:     F    X    F    C    F    F
Row 2:     W    F    F    F    D    R
Row 3:     F    D    F    X    F    F
Row 4:     F    F    C    F    D    F
Row 5:     F    F    F    X    F    R
```

`S`=Start, `F`=Free, `W`=Wind, `D`=Danger, `R`=Rescue target, `C`=Charging station, `X`=Blocked.

### Reward Table

| Event | Reward | Note |
|---|---|---|
| Rescue civilian | +20 | One-time, target disappears |
| Enter danger zone | −10 | Per entry |
| Battery exhausted | −20 | Terminal penalty |
| Reach charging station (by movement) | +5 | Battery refills to MAX |
| Hover on charging station | — | Battery +2 (capped at MAX), no reward |
| Every action | −1 | Step cost |

### Wind Transition Probabilities (30%)

On a wind cell, for directional action `a`:
- P(a executes) = 0.70 + 0.30/4 = **0.775**
- P(each other cardinal direction) = 0.30/4 = **0.075**
- Sum = 0.775 + 3 × 0.075 = 1.0 ✓

---

## Setup & Installation

### Prerequisites

- Python 3.12+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/drone-rescue-value-iteration.git
cd drone-rescue-value-iteration
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the full pipeline from the `src/` directory:

```bash
cd src
python main.py
```

This will:

1. Initialise the `DroneEnv` and verify all Group-176 configuration assertions (grid size, target/charger/danger/blocked counts, battery, step limit, starting position)
2. Print the starting grid layout via `render()`
3. Run Value Iteration to compute `V*(s)` and `π*(s)` over all 2,640 states
4. Print convergence diagnostics (iterations, final delta, runtime)
5. Demonstrate `get_valid_actions()` on the start state
6. Sample and print `V*(s)` values at representative grid positions
7. Simulate one greedy episode following `π*` with a step-by-step log
8. Generate and save four visualisation figures to `../outputs/`:
   - `policy_map.png`
   - `value_heatmap.png`
   - `trajectory.png`
   - `convergence.png`
9. Print a structured analysis covering reward design, wind dynamics, scalability, DP limitations, and a Deep RL extension path

### Expected Results

| Metric | Value |
|---|---|
| Total states | 2,640 |
| Discount γ | 0.9 |
| Threshold θ | 1 × 10⁻³ |
| Runtime | < 5s |
| Trajectory steps | ~20–30 |
| Total reward | ~45–55 |

---

## Documentation

- **`docs/report.md`** — full academic report covering environment design, the Value Iteration algorithm, results analysis, and discussion of DP scalability vs. Deep RL.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computation, state/value arrays |
| `matplotlib` | Policy map, heatmap, trajectory, and convergence plots |

---

## License

This project is provided for educational and academic demonstration purposes.
