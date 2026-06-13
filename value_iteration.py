"""
============================================================
  Value Iteration — Dynamic Programming Solver
  File: value_iteration.py
  Group ID: 176

  Implements the Bellman Optimality Equation:
    V*(s) = max_a  Σ_s' P(s'|s,a) · [R(s,a,s') + γ·V*(s')]

  Stopping condition: max|ΔV| < θ = 10⁻³  (as specified)
============================================================
"""

import time
from environment import DroneEnv, ACTIONS, MAX_STEPS


def value_iteration(env, gamma=0.9, theta=1e-3):
    """
    Compute optimal value function V* and policy π* via Value Iteration.

    Algorithm:
      1. Initialise V(s) = 0 for all states s.
      2. Repeat until max|ΔV| < θ:
           For each non-terminal state s:
             Compute Q(s,a) = Σ_s' P(s'|s,a)[R + γV(s')] for all actions a.
             Update V(s) = max_a Q(s,a).
             Track max change delta = max(delta, |V_new(s) - V_old(s)|).
      3. Extract greedy policy: π*(s) = argmax_a Q(s,a) using converged V*.

    The transition model _get_transitions() handles wind stochasticity
    (30% for Group 176) and returns (prob, next_state, reward, done) tuples.

    Parameters
    ----------
    env   : DroneEnv — Group 176 environment instance
    gamma : float    — discount factor (0.9; standard RL value)
    theta : float    — convergence threshold (1e-3 as specified)

    Returns
    -------
    V      : dict  state → float   — optimal value function V*(s)
    policy : dict  state → int     — optimal policy π*(s) as action index
    info   : dict  diagnostics     — iterations, final_delta, elapsed, etc.
    """
    print("=" * 58)
    print("  VALUE ITERATION — Group ID 176")
    print("=" * 58)

    # Enumerate all valid states for Group 176 (2,640 total)
    states = env.enumerate_states()
    print(f"\n  Grid             : {env.GRID_ROWS}×{env.GRID_COLS}")
    print(f"  Total states     : {len(states)}")
    print(f"  Discount γ       : {gamma}")
    print(f"  Threshold θ      : {theta}")
    print(f"  Wind probability : 30%  (Group ID 176 rule)")
    print(f"  Max steps        : {MAX_STEPS}")
    print(f"  Actions          : {list(ACTIONS.values())}")

    # ── Initialise V(s) = 0 for all states ──────────────────────
    V = {s: 0.0 for s in states}

    iterations = 0
    deltas     = []
    start_time = time.time()

    print("\n  Running iterations …")
    print(f"  {'Iter':>6}  {'Max ΔV':>14}")
    print(f"  {'----':>6}  {'--------':>14}")

    # ── Main Value Iteration loop ────────────────────────────────
    while True:
        delta = 0.0

        for s in states:
            row, col, battery, rescued = s

            # Skip terminal states: V stays at 0 (no future reward possible)
            if battery == 0:
                continue
            if all(rescued):
                continue

            old_v = V[s]

            # ── Compute Q(s,a) for each action and take the max ─
            best_q = float('-inf')
            for a in range(len(ACTIONS)):
                # Get transition distribution from the environment model
                transitions = env._get_transitions(s, a)
                q = 0.0
                for prob, ns, reward, done in transitions:
                    # Bellman update: immediate reward + discounted future value
                    # Terminal states have V = 0 (no future reward)
                    future = V.get(ns, 0.0) if not done else 0.0
                    q += prob * (reward + gamma * future)
                if q > best_q:
                    best_q = q

            # Update value function in place (synchronous update)
            V[s]  = best_q
            delta = max(delta, abs(V[s] - old_v))

        iterations += 1
        deltas.append(delta)

        # Print every iteration as required by assignment
        print(f"  {iterations:>6}  {delta:>14.6f}")

        # ── Check convergence condition: max|ΔV| < θ ──────────
        if delta < theta:
            print(f"\n  ✓ Converged after {iterations} iterations")
            print(f"  ✓ Final max ΔV = {delta:.2e}  (threshold θ = {theta})")
            break

    elapsed = time.time() - start_time
    print(f"  ✓ Runtime: {elapsed:.3f} seconds")

    # ── Extract greedy policy π*(s) = argmax_a Q(s,a) ───────────
    policy = {}
    for s in states:
        row, col, battery, rescued = s

        # Terminal states: assign HOVER as default (no movement needed)
        if battery == 0 or all(rescued):
            policy[s] = 4   # HOVER
            continue

        # Select action with the highest Q value using the converged V*
        best_a = 0
        best_q = float('-inf')
        for a in range(len(ACTIONS)):
            transitions = env._get_transitions(s, a)
            q = 0.0
            for prob, ns, reward, done in transitions:
                future = V.get(ns, 0.0) if not done else 0.0
                q += prob * (reward + gamma * future)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a

    print("  ✓ Optimal policy π* extracted.")
    print("=" * 58)

    return V, policy, {
        'iterations' : iterations,
        'final_delta': deltas[-1],
        'elapsed'    : elapsed,
        'deltas'     : deltas,
        'n_states'   : len(states),
    }
