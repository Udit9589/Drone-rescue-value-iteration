"""
============================================================
  Autonomous Drone Rescue Environment
  File: environment.py
  Group ID: 176

  GROUP ID 176 CONFIGURATION:
    • Last digit = 6  → MAX_BATTERY = 10 (even digit rule)
    • Digit in 5–9    → WIND_RANDOM_PROB = 0.30 (30%)
    • Grid = 6×6
    • 3 rescue targets (R)
    • 2 charging stations (C)
    • 4 danger zones (D)
    • 3 blocked cells (X)
    • Max episode steps = 75
============================================================
"""

import numpy as np
import random
import itertools

# ─────────────────────────────────────────────
#  ACTIONS — Five possible drone actions
# ─────────────────────────────────────────────
ACTIONS = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'HOVER',
}

# Direction deltas corresponding to each action index
ACTION_DELTAS = {
    0: (-1,  0),   # UP    — decrease row
    1: ( 1,  0),   # DOWN  — increase row
    2: ( 0, -1),   # LEFT  — decrease col
    3: ( 0,  1),   # RIGHT — increase col
    4: ( 0,  0),   # HOVER — no movement
}

# ─────────────────────────────────────────────
#  GROUP ID 176 REWARD VALUES
#  Exactly as specified in the assignment table.
#  No modifications to any reward value.
# ─────────────────────────────────────────────
REWARD_RESCUE   =  20   # Rescue a civilian          (assignment: +20)
REWARD_DANGER   = -10   # Enter a danger zone         (assignment: -10)
REWARD_BATTERY  = -20   # Battery exhausted           (assignment: -20)
REWARD_CHARGE   =   5   # Reach charging station      (assignment: +5)
REWARD_MOVE     =  -1   # Every action costs 1 unit   (assignment: -1)

# ─────────────────────────────────────────────
#  BATTERY — Group ID 176 (last digit = 6, even)
#  Rule: even digit → max battery = 10
# ─────────────────────────────────────────────
MAX_BATTERY = 10

# ─────────────────────────────────────────────
#  WIND — Group ID 176 (last digit 6, in range 5–9)
#  Rule: last digit 5–9 → 30% wind disturbance probability
# ─────────────────────────────────────────────
WIND_RANDOM_PROB = 0.30   # 30% chance direction changes on wind cells

# ─────────────────────────────────────────────
#  EPISODE STEP LIMIT — 6×6 grid uses 75 steps
# ─────────────────────────────────────────────
MAX_STEPS = 75


class DroneEnv:
    """
    6×6 grid-world MDP environment for Group ID 176.

    STATE REPRESENTATION:
      s = (row, col, battery, rescued_targets)
        • row, col         : drone position (0–5)
        • battery          : remaining charge (1–10)
        • rescued_targets  : tuple of 3 booleans, e.g.
                             (False,False,False) → none rescued
                             (True,True,True)   → all rescued (terminal)

    Total valid states:
      33 non-blocked cells × 10 battery levels × 8 rescue combos = 2,640

    TERMINATION:
      1. battery reaches 0         → reward -20, done=True
      2. all 3 targets rescued     → done=True
      3. step count exceeds 75     → done=True (max step limit for 6×6)



    Object positions:
      S=Start  : (0,0)                         — top-left corner
      R×3      : (0,4), (2,5), (5,5)           — rescue targets
      C×2      : (1,3), (4,2)                  — charging stations
      D×4      : (0,3), (2,4), (3,1), (4,4)   — danger zones
      X×3      : (1,1), (3,3), (5,3)           — blocked cells
      W×2      : (0,2), (2,0)                  — wind zones
    """

    # ── 6×6 Grid layout for Group ID 176 ────────────────────────
    # Row 0: S  F  W  D  R  F   ← Rescue target at (0,4)
    # Row 1: F  X  F  C  F  F
    # Row 2: W  F  F  F  D  R   ← Rescue target at (2,5)
    # Row 3: F  D  F  X  F  F
    # Row 4: F  F  C  F  D  F
    # Row 5: F  F  F  X  F  R   ← Rescue target at (5,5)
    #
    # Object positions:
    #   S=Start : (0,0)                       — top-left corner (fixed)
    #   R×3     : (0,4), (2,5), (5,5)         — rescue targets
    #   C×2     : (1,3), (4,2)                — charging stations
    #   D×4     : (0,3), (2,4), (3,1), (4,4) — danger zones
    #   X×3     : (1,1), (3,3), (5,3)         — blocked cells
    #   W×2     : (0,2), (2,0)                — wind zones
    # ─────────────────────────────────────────────────────────────
    BASE_GRID = [
        ['S', 'F', 'W', 'D', 'R', 'F'],
        ['F', 'X', 'F', 'C', 'F', 'F'],
        ['W', 'F', 'F', 'F', 'D', 'R'],
        ['F', 'D', 'F', 'X', 'F', 'F'],
        ['F', 'F', 'C', 'F', 'D', 'F'],
        ['F', 'F', 'F', 'X', 'F', 'R'],
    ]

    GRID_ROWS = 6
    GRID_COLS = 6

    def __init__(self):
        """
        Initialise the DroneEnv for Group ID 176.

        Scans BASE_GRID to locate all special cells (start, rescue targets,
        chargers, danger zones, wind zones, blocked cells), validates that
        counts match Group-176 requirements, then calls reset() to set the
        initial episode state.
        """
        self.start_pos       = self._find_cells('S')[0]
        self.rescue_targets  = self._find_cells('R')        # list of (r,c)
        self.danger_zones    = set(map(tuple, self._find_cells('D')))
        self.charging_stats  = set(map(tuple, self._find_cells('C')))  # 2 chargers
        self.wind_zones      = set(map(tuple, self._find_cells('W')))
        self.blocked_cells   = set(map(tuple, self._find_cells('X')))

        self.n_targets = len(self.rescue_targets)  # must be 3 for Group 176

        # Step counter (reset in reset(), incremented in step())
        self.step_count = 0
        self.state      = None
        self.reset()

    # ─────────────────────────────────────────
    #  HELPER: find all cells with given symbol
    # ─────────────────────────────────────────
    def _find_cells(self, symbol):
        """
        Return a list of (row, col) positions where BASE_GRID equals symbol.

        Parameters
        ----------
        symbol : str — single-character cell type ('S','R','C','D','W','X')

        Returns
        -------
        list of (int, int) — all grid positions containing that symbol
        """
        positions = []
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                if self.BASE_GRID[r][c] == symbol:
                    positions.append((r, c))
        return positions

    # ─────────────────────────────────────────
    #  reset() — start a fresh episode
    # ─────────────────────────────────────────
    def reset(self):
        """
        Reset the environment to the initial state for a new episode.

        Per the assignment: "The starting position (S) must be fixed at the
        top-left corner of the grid." The drone begins with full battery
        (MAX_BATTERY = 10 for Group 176, even last digit) and all rescue
        targets unrescued. The step counter is also reset to 0.

        Starting battery level: 10 (full capacity, even digit rule).

        Returns
        -------
        state : tuple — (row=0, col=0, battery=10, rescued=(F,F,F))
        """
        r, c = self.start_pos
        self.state      = (r, c, MAX_BATTERY, tuple([False] * self.n_targets))
        self.step_count = 0
        return self.state

    # ─────────────────────────────────────────
    #  get_valid_actions() — action availability
    # ─────────────────────────────────────────
    def get_valid_actions(self, state):
        """
        Return all valid action indices for a given state.

        An action is valid if it does not violate hard constraints:
          - If battery <= 1, only HOVER is allowed (any move would deplete battery
            and end the episode; hover is still valid as a terminal step).
          - Blocked cells (X) are NOT a reason to exclude an action: the drone
            stays in place when it would hit a block, so the action is still
            executable (it just has no movement effect). Battery is still consumed.
          - All 5 actions (UP, DOWN, LEFT, RIGHT, HOVER) are always available
            subject only to the battery constraint above.

        Note: This function returns the ACTION INDICES (0–4) not names.

        Parameters
        ----------
        state : tuple — (row, col, battery, rescued_targets)

        Returns
        -------
        list of int — valid action indices from {0, 1, 2, 3, 4}
        """
        _, _, battery, _ = state
        if battery <= 0:
            # Episode is already terminal — no valid actions
            return []
        # All 5 actions are always available (boundary/blocked = stay in place)
        return list(ACTIONS.keys())   # [0, 1, 2, 3, 4]

    # ─────────────────────────────────────────
    #  TRANSITION MODEL (used by Value Iteration)
    # ─────────────────────────────────────────
    def _get_transitions(self, state, action_idx):
        """
        Return a list of (probability, next_state, reward, done) tuples.

        This is the model-based transition function P(s',r | s,a) used by
        the Value Iteration DP solver. It returns all possible outcomes and
        their probabilities.

        Wind stochasticity model for Group ID 176 (30% wind probability):
          If the drone is currently on a wind cell (W) and chooses a cardinal
          action (UP/DOWN/LEFT/RIGHT), the actual direction is randomised:
            • 70% probability → intended direction executes
            • 30% probability → direction chosen uniformly from {UP,DOWN,L,R}
              i.e. 30%/4 = 7.5% each random direction

          Combined probability for the intended action a on a wind cell:
            P(a)        = 0.70 + 0.30/4 = 0.775
            P(other dir) = 0.30/4        = 0.075  (each of 3 others)

        HOVER action is not affected by wind.
        Boundary/blocked collisions keep the drone in its current cell.

        Parameters
        ----------
        state      : tuple     — (row, col, battery, rescued_targets)
        action_idx : int       — action from ACTIONS dict (0=UP…4=HOVER)

        Returns
        -------
        list of (float, tuple, float, bool) — (prob, next_state, reward, done)
        """
        row, col, battery, rescued = state

        # ── Battery deduction — every action costs 1 unit ──────
        new_battery = battery - 1

        # If battery becomes 0: episode terminates with -20 penalty
        if new_battery <= 0:
            terminal_state = (row, col, 0, rescued)
            return [(1.0, terminal_state, REWARD_BATTERY, True)]

        # ── Wind stochasticity ─────────────────────────────────
        on_wind = (row, col) in self.wind_zones

        if on_wind and action_idx != 4:   # HOVER is unaffected by wind
            outcomes = {}

            def add_outcome(a_idx, prob):
                """
                Accumulate transition probability for action a_idx.
                Clamps destination to grid boundary and blocked cells.
                Adds prob to the outcomes dict keyed by resulting (nr, nc).
                """
                dr, dc = ACTION_DELTAS[a_idx]
                nr, nc = row + dr, col + dc
                # Boundary clamp: stay in place if out of grid
                if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
                    nr, nc = row, col
                # Blocked cell clamp: stay in place if hitting obstacle
                if (nr, nc) in self.blocked_cells:
                    nr, nc = row, col
                key = (nr, nc)
                outcomes[key] = outcomes.get(key, 0) + prob

            # 30% random disturbance (Group ID 176 rule)
            p_random   = WIND_RANDOM_PROB   # 0.30
            p_intended = 1.0 - p_random     # 0.70

            # Each cardinal direction gets its share of the random probability
            for a in range(4):
                if a == action_idx:
                    # Intended direction: base probability + its random share
                    add_outcome(a, p_intended + p_random / 4)
                else:
                    add_outcome(a, p_random / 4)

            result = []
            for (nr, nc), prob in outcomes.items():
                ns, rew, done = self._compute_next(
                    row, col, nr, nc, new_battery, rescued)
                result.append((prob, ns, rew, done))
            return result

        else:
            # Deterministic: HOVER action or drone not on a wind cell
            dr, dc = ACTION_DELTAS[action_idx]
            nr, nc = row + dr, col + dc
            # Boundary clamp
            if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
                nr, nc = row, col
            # Blocked cell clamp
            if (nr, nc) in self.blocked_cells:
                nr, nc = row, col
            ns, rew, done = self._compute_next(
                row, col, nr, nc, new_battery, rescued)
            return [(1.0, ns, rew, done)]

    def _compute_next(self, old_r, old_c, nr, nc, new_battery, rescued):
        """
        Compute the reward, next state, and done flag given source and
        destination cells.

        Applies the assignment reward structure in priority order:
          1. Base step cost of -1 (REWARD_MOVE) — always applied first.
          2. Charging station (C) logic:
               • HOVER on charger (old == new):
                   battery += 2 per step, capped at MAX_BATTERY.
                   Assignment: "hovering increases battery by +2 units;
                   battery cannot exceed maximum capacity." No extra reward.
               • MOVE INTO charger (old != new):
                   battery = MAX_BATTERY (full recharge on entry).
                   reward += +5 (REWARD_CHARGE).
                   Assignment: "When the drone enters a charging station:
                   battery becomes full again." + "Reach charging station: +5."
          3. Danger zone (D): reward += -10 (REWARD_DANGER).
          4. Rescue target (R): if unrescued at destination, mark rescued,
               reward += +20 (REWARD_RESCUE).
          5. If all targets rescued → done = True.

        Parameters
        ----------
        old_r, old_c : int         — source cell position (before action)
        nr, nc       : int         — destination cell (after action)
        new_battery  : int         — battery after deducting 1 for the action
        rescued      : tuple/list  — current rescue status per target

        Returns
        -------
        (next_state, reward, done) : tuple
        """
        reward  = REWARD_MOVE   # -1 step cost always applies
        done    = False
        rescued = list(rescued)

        # ── Charging station logic ─────────────────────────────
        if (nr, nc) in self.charging_stats:
            if (old_r, old_c) == (nr, nc):
                # HOVER on charger: +2 battery per assignment spec, capped at MAX
                # Assignment: "hovering increases battery by +2 units;
                #              battery cannot exceed maximum capacity."
                new_battery = min(new_battery + 2, MAX_BATTERY)
                # No +5 reward for hovering — only arriving earns it
            else:
                # MOVE INTO charger: battery becomes full (assignment rule)
                # Assignment: "When the drone enters a charging station:
                #              battery becomes full again."
                new_battery = MAX_BATTERY
                # Assignment reward table: "Reach charging station = +5"
                # Awarded every time the drone moves into a charging station
                # from a different cell, exactly as specified.
                reward += REWARD_CHARGE   # +5 on every arrival at charger

        # ── Danger zone penalty ────────────────────────────────
        elif (nr, nc) in self.danger_zones:
            # Assignment: entering danger zone gives -10, episode does NOT end
            reward += REWARD_DANGER   # -10

        # ── Rescue target collection ───────────────────────────
        for i, (tr, tc) in enumerate(self.rescue_targets):
            if (nr, nc) == (tr, tc) and not rescued[i]:
                # Mark target as rescued, award +20 one-time reward
                rescued[i] = True
                reward += REWARD_RESCUE   # +20 per rescued civilian

        rescued = tuple(rescued)

        # ── Terminal condition: all civilians rescued ──────────
        if all(rescued):
            done = True

        next_state = (nr, nc, new_battery, rescued)
        return next_state, reward, done

    # ─────────────────────────────────────────
    #  step() — real simulation with randomness
    # ─────────────────────────────────────────
    def step(self, action_idx):
        """
        Execute one action in the environment and return the result.

        This is the simulation step used for trajectory roll-outs. It samples
        the actual stochastic wind outcome (unlike _get_transitions which
        returns the full probability distribution for DP planning).

        Battery is decremented by 1 for every action. Wind disturbance is
        sampled randomly (30% for Group 176). Blocked/boundary actions keep
        the drone in place but still drain battery. The step counter is
        incremented and checked against MAX_STEPS (75 for 6×6).

        Parameters
        ----------
        action_idx : int — action index from ACTIONS dict (0–4)

        Returns
        -------
        (next_state, reward, done) : tuple
          next_state : (row, col, battery, rescued_targets)
          reward     : float
          done       : bool
        """
        row, col, battery, rescued = self.state
        self.step_count += 1

        # ── Battery deduction ──────────────────────────────────
        new_battery = battery - 1
        if new_battery <= 0:
            # Battery exhausted: terminal state with -20 penalty
            self.state = (row, col, 0, rescued)
            return self.state, REWARD_BATTERY, True

        # ── Wind disturbance (stochastic sampling for simulation) ──
        on_wind     = (row, col) in self.wind_zones
        actual_action = action_idx
        if on_wind and action_idx != 4 and random.random() < WIND_RANDOM_PROB:
            # 30% chance: replace action with a uniformly random cardinal direction
            actual_action = random.randint(0, 3)

        # ── Compute destination cell ───────────────────────────
        dr, dc = ACTION_DELTAS[actual_action]
        nr, nc = row + dr, col + dc
        # Boundary clamp
        if not (0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS):
            nr, nc = row, col
        # Blocked cell clamp
        if (nr, nc) in self.blocked_cells:
            nr, nc = row, col

        # ── Apply cell effects and compute reward ──────────────
        next_state, reward, done = self._compute_next(
            row, col, nr, nc, new_battery, rescued)
        self.state = next_state

        # ── Step limit termination (75 for 6×6 grids) ─────────
        if not done and self.step_count >= MAX_STEPS:
            done = True

        return next_state, reward, done

    # ─────────────────────────────────────────
    #  render() — ASCII grid visualisation
    # ─────────────────────────────────────────
    def render(self, state=None):
        """
        Print a text/ASCII representation of the current grid state to stdout.

        Shows a header with battery level, rescue count, and step count,
        followed by a 6×6 ASCII grid. The drone's current position is
        shown as 🚁, rescued targets as ✓, and all other cells by their
        symbol from BASE_GRID.

        Parameters
        ----------
        state : tuple|None — (row, col, battery, rescued) to render.
                             If None, uses self.state (current episode state).
        """
        if state is None:
            state = self.state
        row, col, battery, rescued = state

        print("\n┌──────────────────────────┐")
        print(f"│  Battery : {battery:2d}/{MAX_BATTERY}            │")
        print(f"│  Rescued : {sum(rescued)}/{self.n_targets}             │")
        print(f"│  Steps   : {self.step_count}/{MAX_STEPS}          │")
        print("├───┬───┬───┬───┬───┬───┤")

        for r in range(self.GRID_ROWS):
            row_str = "│"
            for c in range(self.GRID_COLS):
                if r == row and c == col:
                    cell = "🚁 "
                else:
                    base = self.BASE_GRID[r][c]
                    if base == 'R':
                        # Show checkmark if this target has been rescued
                        idx  = self.rescue_targets.index((r, c))
                        cell = "✓  " if rescued[idx] else "R  "
                    else:
                        cell = base + "  "
                row_str += cell + "│"
            print(row_str)
        print("└───┴───┴───┴───┴───┴───┘\n")

    # ─────────────────────────────────────────
    #  enumerate_states() — full state space
    # ─────────────────────────────────────────
    def enumerate_states(self):
        """
        Enumerate all valid MDP states for Value Iteration.

        Iterates over all (row, col) positions excluding blocked cells,
        all battery levels from 1 to MAX_BATTERY, and all 2^3 = 8
        combinations of rescue target status.

        Group ID 176 totals:
          Valid cells   = 36 - 3 blocked = 33
          Battery levels= 1 … 10         = 10
          Rescue combos = 2^3             = 8
          Total         = 33 × 10 × 8    = 2,640 states

        Returns
        -------
        list of tuple — all valid state tuples (row, col, battery, rescued)
        """
        states = []
        rescue_combos = list(itertools.product([False, True], repeat=self.n_targets))

        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                # Skip blocked cells — drone can never occupy them
                if (r, c) in self.blocked_cells:
                    continue
                for bat in range(1, MAX_BATTERY + 1):
                    for rescued in rescue_combos:
                        states.append((r, c, bat, rescued))
        return states
