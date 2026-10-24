"""Tabular Q-learning for the warehouse bot: Q-table, epsilon-greedy, training loop.

Run:  python qlearning.py
"""
import random
import numpy as np

from env import WarehouseEnv, STEP_REWARD

# ---- settings (hyperparameters) -----------------------------------------
ALPHA = 0.1                        # learning rate
GAMMA = 0.95                       # discount
EPS_START = 1.0                    # epsilon at episode 0
EPS_END = 0.01                     # epsilon floor

N_EPISODES = 30000                 # training length
EPS_DECAY_EPISODES = 22500         # epsilon falls over the first 75% of the run
REPORT_EVERY = 3000                # print one progress line per this many episodes

SEED = 0
SAVE_FILE = "q_table.npy"


def epsilon_for(episode, eps_start=EPS_START, eps_end=EPS_END, decay_episodes=EPS_DECAY_EPISODES):
    """Epsilon for a given episode: falls in a straight line, then stays at eps_end."""
    fraction_done = episode / decay_episodes
    epsilon = eps_start - (eps_start - eps_end) * fraction_done
    return max(eps_end, epsilon)


def choose_action(q_table, state, epsilon):
    """Epsilon-greedy: explore with probability epsilon, otherwise take the best-known action."""
    if random.random() < epsilon:
        return random.randrange(WarehouseEnv.n_actions)
    q_values = q_table[state]
    best_actions = np.flatnonzero(q_values == q_values.max())
    return int(random.choice(best_actions))


def train(alpha=ALPHA, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
          eps_decay_episodes=EPS_DECAY_EPISODES, step_reward=STEP_REWARD, seed=SEED,
          n_episodes=N_EPISODES, snapshot_at=(), snapshots=None, record_at=(), recordings=None):
    """Run the whole training. Returns the Q-table and the return of every episode."""
    random.seed(seed)
    env = WarehouseEnv(step_reward=step_reward)
    q_table = np.zeros((env.n_states, env.n_actions))
    episode_returns = []

    for episode in range(n_episodes):
        if snapshots is not None and episode in snapshot_at:
            snapshots[episode] = q_table.copy()
        epsilon = epsilon_for(episode, eps_start, eps_end, eps_decay_episodes)
        state = env.reset()
        total = 0
        done = False
        episode_actions = [] if (recordings is not None and episode in record_at) else None

        while not done:
            action = choose_action(q_table, state, epsilon)
            if episode_actions is not None:
                episode_actions.append(action)
            next_state, reward, done = env.step(action)

            if env.delivered:
                target = reward                     # true end: nothing comes after it
            else:
                target = reward + gamma * q_table[next_state].max()

            q_table[state, action] += alpha * (target - q_table[state, action])

            state = next_state
            total += reward

        if episode_actions is not None:
            recordings[episode] = episode_actions
        episode_returns.append(total)

    if snapshots is not None and n_episodes in snapshot_at:
        snapshots[n_episodes] = q_table.copy()      # the finished table

    return q_table, np.array(episode_returns)


def main():
    random.seed(SEED)
    q_table, returns = train()

    print(f"Grid {WarehouseEnv.SIZE}x{WarehouseEnv.SIZE}. Trained for {N_EPISODES} episodes "
          f"(alpha={ALPHA}, gamma={GAMMA}, epsilon {EPS_START} -> {EPS_END} over {EPS_DECAY_EPISODES} episodes)")
    print("Average return per block of episodes (training, with exploration):")
    for start in range(0, N_EPISODES, REPORT_EVERY):
        block = returns[start:start + REPORT_EVERY]
        end = start + len(block) - 1
        print(f"  episodes {start:5d}-{end:5d}: average {block.mean():9.1f}   "
              f"epsilon at end {epsilon_for(end):.2f}")

    print(f"Q-table entries changed from zero: {np.count_nonzero(q_table)} of {q_table.size}")
    print(f"Q-value range: {q_table.min():.1f} to {q_table.max():.1f}")
    np.save(SAVE_FILE, q_table)
    print(f"Saved Q-table to {SAVE_FILE}")


if __name__ == "__main__":
    main()
