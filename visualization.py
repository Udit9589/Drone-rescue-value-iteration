"""
============================================================
  Visualization Module
  File: visualization.py
  Group ID: 176 — 6×6 grid

  This module provides all visualization utilities for the
  Drone Rescue DP project. It generates:
    - Policy map     : arrow/symbol overlaid on grid
    - Value heatmap  : colour-coded V*(s) across grid cells
    - Trajectory plot: step-numbered drone path overlay
    - Convergence    : max ΔV vs iteration (log scale)
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from environment import DroneEnv, ACTIONS, ACTION_DELTAS, MAX_BATTERY, MAX_STEPS

# Arrow symbol for each action index (0-4)
ARROW_SYMBOLS = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '●'}

# Colour scheme for each cell type in the grid
CELL_COLORS = {
    'S': '#90EE90',   # Start     — light green
    'F': '#F5F5F5',   # Free      — off-white
    'D': '#FF6B6B',   # Danger    — red
    'R': '#FFD700',   # Rescue    — gold
    'C': '#87CEEB',   # Charger   — sky blue
    'W': '#DDA0DD',   # Wind      — plum
    'X': '#2C2C2C',   # Blocked   — dark grey
}


def plot_policy(env, policy, battery=MAX_BATTERY,
                rescued=None, save_path=None):
    """
    Visualise the optimal policy π* as action arrows on the 6×6 grid.

    For a fixed (battery, rescued_targets) slice of the state space,
    each non-blocked cell is annotated with the optimal action symbol:
      ↑ ↓ ← → for movement, ● for HOVER.

    Parameters
    ----------
    env       : DroneEnv   — the Group-176 environment instance
    policy    : dict       — state → action_index mapping from value iteration
    battery   : int        — battery level to slice (default MAX_BATTERY=10)
    rescued   : tuple      — rescue status tuple to slice (default all False)
    save_path : str|None   — if given, saves figure to this path
    """
    if rescued is None:
        rescued = tuple([False] * env.n_targets)

    rows, cols = env.GRID_ROWS, env.GRID_COLS
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw each cell with its type colour and label
    for r in range(rows):
        for c in range(cols):
            base  = env.BASE_GRID[r][c]
            color = CELL_COLORS.get(base, '#F5F5F5')
            rect  = plt.Rectangle(
                [c, rows - 1 - r], 1, 1,
                facecolor=color, edgecolor='black', lw=1.5)
            ax.add_patch(rect)
            ax.text(c + 0.07, rows - 1 - r + 0.82, base,
                    fontsize=8, color='#333', weight='bold')

            # Overlay policy arrow for this (position, battery, rescued) state
            state = (r, c, battery, rescued)
            if policy.get(state) is not None:
                sym = ARROW_SYMBOLS[policy[state]]
                ax.text(c + 0.5, rows - 1 - r + 0.4, sym,
                        fontsize=20, ha='center', va='center',
                        color='#003366', weight='bold')

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels(range(cols))
    ax.set_yticklabels(range(rows - 1, -1, -1))
    ax.set_title(
        f'Optimal Policy π* — Group ID 176\n'
        f'Battery={battery}, Rescued={rescued}',
        fontsize=13, weight='bold')

    patches = [mpatches.Patch(color=v, label=k) for k, v in CELL_COLORS.items()]
    ax.legend(handles=patches, loc='upper right',
              bbox_to_anchor=(1.16, 1.0), fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_value_heatmap(env, V, battery=MAX_BATTERY,
                       rescued=None, save_path=None):
    """
    Generate a colour heatmap of V*(s) across all grid positions.

    Fixes battery and rescued_targets to create a 2-D slice of the
    (3-D+) value function. Blocked cells (X) appear dark with an 'X'
    label; all other cells display their numeric V* value.

    The colour gradient runs from dark purple (low/negative value) through
    blue/cyan to yellow/orange (high positive value), giving an intuitive
    view of which regions the drone should prefer.

    Parameters
    ----------
    env       : DroneEnv   — Group-176 environment instance
    V         : dict       — state → float value function from value iteration
    battery   : int        — battery level to slice (default MAX_BATTERY=10)
    rescued   : tuple      — rescue status tuple to slice (default all False)
    save_path : str|None   — if given, saves figure to this path
    """
    if rescued is None:
        rescued = tuple([False] * env.n_targets)

    rows, cols = env.GRID_ROWS, env.GRID_COLS
    grid_v = np.full((rows, cols), np.nan)

    # Fill grid with V*(s) for non-blocked cells
    for r in range(rows):
        for c in range(cols):
            s = (r, c, battery, rescued)
            if s in V:
                grid_v[r][c] = V[s]

    fig, ax = plt.subplots(figsize=(8, 7))
    # Custom diverging colourmap: dark → purple → blue → cyan → yellow → orange
    cmap = LinearSegmentedColormap.from_list(
        'drone', ['#0d0221', '#4b0082', '#0000cc',
                  '#0099ff', '#00ff99', '#ffff00', '#ff4400'])

    masked = np.ma.masked_invalid(grid_v)
    im = ax.imshow(masked, cmap=cmap, aspect='equal')
    plt.colorbar(im, ax=ax, label='V*(s)')

    # Annotate each cell with its V* value and cell-type label
    for r in range(rows):
        for c in range(cols):
            base = env.BASE_GRID[r][c]
            val  = grid_v[r][c]
            if np.isnan(val):
                ax.text(c, r, 'X', ha='center', va='center',
                        fontsize=13, color='white', weight='bold')
            else:
                ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                        fontsize=9, color='white', weight='bold')
                ax.text(c - 0.42, r - 0.38, base,
                        fontsize=7, color='yellow')

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title(
        f'Value Function Heatmap — Group ID 176\n'
        f'Battery={battery}, Rescued={rescued}',
        fontsize=13, weight='bold')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def simulate_trajectory(env, policy, max_steps=MAX_STEPS):
    """
    Run one episode using the optimal policy and record each transition.

    The drone starts from the environment's initial state (via env.reset())
    and follows the greedy policy at each step. If a state is not in the
    policy dictionary, HOVER (action 4) is used as a safe default.

    Each step is printed with: position, battery, action chosen, step reward,
    cumulative reward, and rescue count.

    Parameters
    ----------
    env       : DroneEnv — Group-176 environment instance (will be reset)
    policy    : dict     — state → action_index mapping from value iteration
    max_steps : int      — maximum number of steps to simulate (default 75)

    Returns
    -------
    trajectory : list of (state, action, reward) tuples — one entry per step
    """
    state = env.reset()
    trajectory = []
    total_reward = 0

    print(f"\n  ── Optimal Trajectory (max {max_steps} steps) ──")
    print(f"  {'Step':>4}  {'Pos':>6}  {'Bat':>4}  "
          f"{'Action':>6}  {'Reward':>7}  {'Total':>8}  Rescued")
    print("  " + "-" * 58)

    for step in range(max_steps):
        action = policy.get(state, 4)   # default HOVER if state not found
        r, c, bat, rescued = state
        action_name = ACTIONS[action]
        next_state, reward, done = env.step(action)
        total_reward += reward

        print(f"  {step:>4}  ({r},{c})  {bat:>4}  "
              f"{action_name:>6}  {reward:>7.1f}  {total_reward:>8.1f}"
              f"  {sum(rescued)}/{env.n_targets}")

        trajectory.append((state, action, reward))
        state = next_state

        if done:
            r2, c2, bat2, rescued2 = state
            print(f"\n  → Episode ended at step {step + 1}")
            print(f"  → Final: pos=({r2},{c2}), battery={bat2}, "
                  f"rescued={sum(rescued2)}/{env.n_targets}")
            print(f"  → Total reward: {total_reward:.1f}")
            break
    else:
        print(f"\n  → Step limit ({max_steps}) reached.")
        print(f"  → Total reward: {total_reward:.1f}")

    return trajectory


def plot_trajectory(env, trajectory, save_path=None):
    """
    Plot the drone's step-by-step trajectory overlaid on the 6×6 grid.

    Each grid cell is coloured by type. The drone's path is drawn as a green
    line connecting visited cell centres. Each visited position is labelled with
    its step number (0 = start). If the same cell is visited multiple times the
    labels are nudged outward so all step numbers remain readable.

    Parameters
    ----------
    env        : DroneEnv   — Group-176 environment instance
    trajectory : list       — list of (state, action, reward) from simulate_trajectory
    save_path  : str|None   — if given, saves figure to this path
    """
    rows, cols = env.GRID_ROWS, env.GRID_COLS
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw grid background
    for r in range(rows):
        for c in range(cols):
            base  = env.BASE_GRID[r][c]
            color = CELL_COLORS.get(base, '#F5F5F5')
            rect  = plt.Rectangle(
                [c, rows - 1 - r], 1, 1,
                facecolor=color, edgecolor='black', lw=1.5)
            ax.add_patch(rect)
            ax.text(c + 0.07, rows - 1 - r + 0.82, base,
                    fontsize=8, color='#333', weight='bold')

    # Build ordered list of (step_number, row, col) from trajectory states
    # trajectory[i] = (state_before_step_i, action, reward)
    from collections import defaultdict

    positions = []
    for i, (state, action, reward) in enumerate(trajectory):
        r, c, bat, rescued = state
        positions.append((i, r, c))

    # Append the final destination cell (last state after last action)
    if trajectory:
        last_state, last_action, _ = trajectory[-1]
        lr, lc = last_state[0], last_state[1]
        dr, dc = ACTION_DELTAS[last_action]
        nr, nc = lr + dr, lc + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = lr, lc
        if (nr, nc) in env.blocked_cells:
            nr, nc = lr, lc
        positions.append((len(trajectory), nr, nc))

    # Small nudge offsets so step labels don't overlap when revisiting a cell
    NUDGE = [
        ( 0.00,  0.00),   # 1st visit: centre
        ( 0.27,  0.00),   # 2nd visit: right
        (-0.27,  0.00),   # 3rd visit: left
        ( 0.00,  0.27),   # 4th visit: up
        ( 0.00, -0.27),   # 5th visit: down
        ( 0.22,  0.22),   # 6th+
        (-0.22,  0.22),
    ]
    visit_count = defaultdict(int)

    xs, ys = [], []
    for step_num, r, c in positions:
        cx = c + 0.5
        cy = rows - 1 - r + 0.5
        xs.append(cx)
        ys.append(cy)

        visit = visit_count[(r, c)]
        visit_count[(r, c)] += 1
        ox, oy = NUDGE[min(visit, len(NUDGE) - 1)]

        ax.text(cx + ox, cy + oy, str(step_num),
                fontsize=7, ha='center', va='center',
                color='navy', weight='bold',
                bbox=dict(boxstyle='circle,pad=0.13', facecolor='white',
                          edgecolor='navy', alpha=0.90), zorder=4)

    # Draw path line and start/end markers
    ax.plot(xs, ys, 'g-', linewidth=2, alpha=0.55, zorder=1)
    if xs:
        ax.plot(xs[0], ys[0], 'go', markersize=14, label='Start', zorder=5)
        ax.plot(xs[-1], ys[-1], 'rs', markersize=14, label='End',   zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_title('Optimal Drone Trajectory — Group ID 176',
                 fontsize=13, weight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_convergence(info, save_path=None):
    """
    Plot the Value Iteration convergence curve (max ΔV vs iteration).

    The y-axis uses a logarithmic scale to make the exponential convergence
    clearly visible. A horizontal dashed red line marks the stopping threshold
    θ = 1e-3. The curve should reach the threshold line at the final iteration.

    Parameters
    ----------
    info      : dict   — diagnostics dict returned by value_iteration()
                         Must contain key 'deltas' (list of per-iteration max ΔV)
    save_path : str|None — if given, saves figure to this path
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(info['deltas'], color='#003366', linewidth=2)
    ax.axhline(y=1e-3, color='red', linestyle='--',
               linewidth=1.5, label='θ = 1e-3')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Max ΔV (log scale)', fontsize=12)
    ax.set_title('Value Iteration Convergence — Group ID 176',
                 fontsize=13, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
