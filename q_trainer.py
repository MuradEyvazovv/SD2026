"""
=============================================================
Q-LEARNING TRAINER - MARL Testbed (FINAL)
Team 4 Senior Design 2026

STATE REPRESENTATION:
  (agent_row, agent_col, dir_to_nearest_unexplored, dir_to_ally)
  
  This richer state lets the Q-table learn:
  - "When unexplored cells are to the North, go North"
  - "When ally is nearby, move away"
  
  Without this, Q-table only learns position -> action
  which doesn't generalize to new exploration scenarios.

DIRECTIONS: N, S, E, W, NE, NW, SE, SW, NONE = 9 options
STATE SPACE: 63 * 9 * 9 = 5,103 states (manageable)

HOW TO RUN:
  python3 q_trainer.py

OUTPUT:
  q_tables.pkl           <- loaded by ml_decide.py
  q_training_results.png <- show in presentation
=============================================================
"""

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# GRID SETUP
# =============================================================
GRID_ROWS   = 7
GRID_COLS   = 9
TOTAL_CELLS = GRID_ROWS * GRID_COLS  # 63

ACTIONS = ['Forward', 'Backward', 'Left', 'Right']
ACTION_DELTAS = {
    'Forward':  (-1,  0),
    'Backward': ( 1,  0),
    'Left':     ( 0, -1),
    'Right':    ( 0,  1),
}

# Direction encoding
DIRS = ['N','NE','E','SE','S','SW','W','NW','NONE']

def encode_direction(from_pos, to_pos):
    """Encode direction from one position to another as compass."""
    if to_pos is None:
        return 'NONE'
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    if dr == 0 and dc == 0:
        return 'NONE'
    angle = np.degrees(np.arctan2(dc, -dr)) % 360
    idx = int((angle + 22.5) / 45) % 8
    return ['N','NE','E','SE','S','SW','W','NW'][idx]

def nearest_unexplored(pos, explored):
    """Find nearest unexplored cell using Manhattan distance."""
    best     = None
    best_d   = float('inf')
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if (r,c) not in explored:
                d = abs(r-pos[0]) + abs(c-pos[1])
                if d < best_d:
                    best_d = d
                    best   = (r,c)
    return best

def get_state(pos, explored, ally_positions):
    """
    State = (row, col, dir_to_nearest_unexplored, dir_to_nearest_ally)
    """
    nearest_unexp = nearest_unexplored(pos, explored)
    dir_unexp     = encode_direction(pos, nearest_unexp)

    if ally_positions:
        nearest_ally = min(ally_positions,
                           key=lambda p: abs(p[0]-pos[0])+abs(p[1]-pos[1]))
        dir_ally = encode_direction(pos, nearest_ally)
    else:
        dir_ally = 'NONE'

    return (pos[0], pos[1], dir_unexp, dir_ally)

# =============================================================
# Q-LEARNING PARAMETERS
# =============================================================
NUM_EPISODES = 800
MAX_STEPS    = 250
ALPHA        = 0.25
GAMMA        = 0.85
EPS_START    = 1.0
EPS_END      = 0.05
EPS_DECAY    = 0.993
NUM_AGENTS   = 2

# Rewards
R_TARGET   =  100
R_NEW_CELL =   15
R_SPREAD   =    5
R_STEP     =   -1
R_REVISIT  =   -3

# =============================================================
# Q-TABLE: sparse dict (state -> {action -> value})
# =============================================================
def get_q(q_table, state, action):
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in ACTIONS}
    return q_table[state][action]

def set_q(q_table, state, action, value):
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in ACTIONS}
    q_table[state][action] = value

def best_q(q_table, state, valid_actions):
    if state not in q_table:
        return 0.0, random.choice(valid_actions)
    vals   = {a: q_table[state][a] for a in valid_actions}
    best_a = max(vals, key=vals.get)
    return vals[best_a], best_a

def get_valid_actions(pos):
    valid = []
    for action, (dr, dc) in ACTION_DELTAS.items():
        nr, nc = pos[0]+dr, pos[1]+dc
        if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
            valid.append(action)
    return valid

def select_action(q_table, state, epsilon, valid_actions):
    if not valid_actions:
        return random.choice(ACTIONS)
    if random.random() < epsilon:
        return random.choice(valid_actions)
    _, action = best_q(q_table, state, valid_actions)
    return action

# =============================================================
# COMPUTE REWARD
# =============================================================
def compute_reward(new_pos, old_pos, explored,
                   all_positions, agent_id, target_pos):
    if new_pos == target_pos:
        return R_TARGET

    reward = R_STEP

    if new_pos not in explored:
        reward += R_NEW_CELL
    else:
        reward += R_REVISIT

    # Spread reward
    others = [p for i,p in enumerate(all_positions) if i != agent_id]
    if others:
        old_d = min(abs(old_pos[0]-p[0])+abs(old_pos[1]-p[1])
                    for p in others)
        new_d = min(abs(new_pos[0]-p[0])+abs(new_pos[1]-p[1])
                    for p in others)
        if new_d > old_d:
            reward += R_SPREAD

    return reward

# =============================================================
# RUN ONE EPISODE
# =============================================================
def run_episode(q_tables, epsilon, num_agents):
    all_cells  = [(r,c) for r in range(GRID_ROWS)
                         for c in range(GRID_COLS)]
    # Randomize target - agents don't know where it is
    target_pos = random.choice(all_cells)
    non_target = [c for c in all_cells if c != target_pos]
    positions  = random.sample(non_target, num_agents)
    explored   = set(positions)

    total_rewards = [0.0] * num_agents
    target_found  = False
    steps         = 0

    for step in range(MAX_STEPS):
        steps += 1
        snap = positions[:]  # position snapshot this step

        for i in range(num_agents):
            pos   = positions[i]
            allies = [p for j,p in enumerate(snap) if j != i]
            state  = get_state(pos, explored, allies)
            valid  = get_valid_actions(pos)
            action = select_action(q_tables[i], state, epsilon, valid)

            dr, dc  = ACTION_DELTAS[action]
            new_pos = (pos[0]+dr, pos[1]+dc)
            if not (0 <= new_pos[0] < GRID_ROWS and
                    0 <= new_pos[1] < GRID_COLS):
                continue

            reward = compute_reward(new_pos, pos, explored,
                                    snap, i, target_pos)
            total_rewards[i] += reward
            explored.add(new_pos)

            # Bellman update
            new_allies     = [p for j,p in enumerate(snap) if j != i]
            new_explored   = explored | {new_pos}
            next_state     = get_state(new_pos, new_explored, new_allies)
            next_valid     = get_valid_actions(new_pos)
            next_q_val, _  = best_q(q_tables[i], next_state, next_valid)
            old_q_val      = get_q(q_tables[i], state, action)
            new_q_val      = (old_q_val +
                              ALPHA*(reward + GAMMA*next_q_val - old_q_val))
            set_q(q_tables[i], state, action, new_q_val)

            positions[i] = new_pos
            if new_pos == target_pos:
                target_found = True

        if target_found:
            break

    return total_rewards, len(explored), steps, target_found

# =============================================================
# SAVE BEST Q-TABLE (not final — best performing)
# =============================================================
def save_qtables(q_tables, num_agents):
    data = {
        'q_tables':   q_tables,
        'num_agents': num_agents,
        'grid_rows':  GRID_ROWS,
        'grid_cols':  GRID_COLS,
        'actions':    ACTIONS,
        'state_type': 'rich',  # flag for ml_decide.py
    }
    with open('q_tables.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(f"\n[SAVED] q_tables.pkl")
    print(f"  Agents     : {num_agents}")
    print(f"  State type : (row, col, dir_unexplored, dir_ally)")
    print(f"  Actions    : {ACTIONS}")

# =============================================================
# PLOT TRAINING RESULTS
# =============================================================
def plot_training(steps_hist, cov_hist, rew_hist, found_hist):
    w   = 50
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        'Q-Learning Training Results — MARL Testbed\n'
        'Team 4 Senior Design 2025-2026\n'
        'State: (position, direction to unexplored, direction to ally)',
        fontsize=12, fontweight='bold'
    )

    def smooth(data, w):
        s = np.convolve(data, np.ones(w)/w, mode='valid')
        return list(range(w-1, len(data))), list(s)

    datasets = [
        (axes[0][0], steps_hist,   'steelblue',  'Steps per Episode',
         'Steps', 'Lower = More Efficient'),
        (axes[0][1], cov_hist,     'green',       'Coverage per Episode',
         'Coverage (%)', 'Higher = Better Exploration'),
        (axes[1][1], rew_hist,     'darkorange',  'Total Reward per Episode',
         'Reward', 'Higher = Better Policy'),
    ]

    for ax, data, color, title, ylabel, subtitle in datasets:
        ax.plot(data, color=color, alpha=0.2, linewidth=1)
        x, s = smooth(data, w)
        ax.plot(x, s, color=color, linewidth=2.5,
                label=f'Smoothed (w={w})')
        early = np.mean(data[:100])
        late  = np.mean(data[-100:])
        ax.axhline(early, color='red',   linestyle='--',
                   alpha=0.6, label=f'Early: {early:.1f}')
        ax.axhline(late,  color='green', linestyle='--',
                   alpha=0.6, label=f'Late: {late:.1f}')
        ax.set_title(f'{title}\n({subtitle})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Discovery rate
    ax = axes[1][0]
    window_rate = [np.mean(found_hist[max(0,i-50):i+1])*100
                   for i in range(len(found_hist))]
    ax.plot(window_rate, color='purple', linewidth=2.5)
    ax.fill_between(range(len(window_rate)),
                    window_rate, alpha=0.15, color='purple')
    ax.set_title('Target Discovery Rate\n(% of Episodes, 50-ep window)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Discovery Rate (%)')
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('q_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[SAVED] q_training_results.png")

# =============================================================
# MAIN
# =============================================================
if __name__ == '__main__':
    print("="*60)
    print("Q-LEARNING TRAINER - MARL Testbed")
    print("Team 4 Senior Design 2025-2026")
    print(f"Grid: {GRID_ROWS}×{GRID_COLS} | Agents: {NUM_AGENTS}")
    print(f"State: (pos, dir_to_unexplored, dir_to_ally)")
    print(f"Episodes: {NUM_EPISODES} | Target: RANDOMIZED each episode")
    print("="*60 + "\n")

    random.seed(42)
    np.random.seed(42)

    q_tables  = [{} for _ in range(NUM_AGENTS)]
    epsilon   = EPS_START

    steps_hist = []
    cov_hist   = []
    rew_hist   = []
    found_hist = []

    best_cov      = 0
    best_qtables  = None

    for ep in range(NUM_EPISODES):
        rewards, coverage, steps, found = run_episode(
            q_tables, epsilon, NUM_AGENTS
        )
        steps_hist.append(steps)
        cov_hist.append(coverage / TOTAL_CELLS * 100)
        rew_hist.append(sum(rewards))
        found_hist.append(1 if found else 0)

        # Save best performing Q-tables
        avg_cov = np.mean(cov_hist[-10:]) if len(cov_hist) >= 10 else 0
        if avg_cov > best_cov:
            best_cov     = avg_cov
            best_qtables = [dict(q) for q in q_tables]

        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if (ep+1) % 100 == 0:
            w = 100
            print(f"Ep {ep+1:5d}/{NUM_EPISODES} | "
                  f"Steps: {np.mean(steps_hist[-w:]):6.1f} | "
                  f"Coverage: {np.mean(cov_hist[-w:]):5.1f}% | "
                  f"Found: {np.sum(found_hist[-w:])}/{w} | "
                  f"ε={epsilon:.3f}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"  Best avg coverage  : {best_cov:.1f}%")
    print(f"  Final avg steps    : {np.mean(steps_hist[-100:]):.1f}")
    print(f"  Target found rate  : "
          f"{np.sum(found_hist[-100:])/100*100:.0f}%")
    print(f"{'='*60}")

    # Save best Q-tables (not final)
    save_qtables(best_qtables, NUM_AGENTS)
    plot_training(steps_hist, cov_hist, rew_hist, found_hist)

    print("\n[READY] q_tables.pkl is loaded by ml_decide.py")
