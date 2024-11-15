import csv
import numpy as np
from Bandit import EpsilonGreedy, ThompsonSampling, Visualization

n_bandits = 4
bandit_rewards = [1, 2, 3, 4]
n_trials = 20000
epsilon_decay = 0.99

def run_experiment(algorithm, n_trials):
    rewards = []
    cumulative_rewards = 0
    regret = []
    max_reward = max(bandit_rewards) * n_trials

    for _ in range(n_trials):
        bandit = algorithm.pull()
        reward = bandit_rewards[bandit] if np.random.rand() < 0.5 else 0
        algorithm.update(bandit, reward)
        
        rewards.append(reward)
        cumulative_rewards += reward
        regret.append(max_reward - cumulative_rewards)

    return rewards, regret, cumulative_rewards


epsilon_greedy = EpsilonGreedy(n_bandits=n_bandits, epsilon_decay=epsilon_decay)
thompson_sampling = ThompsonSampling(n_bandits=n_bandits)


eg_rewards, eg_regret, eg_cumulative_reward = run_experiment(epsilon_greedy, n_trials)
ts_rewards, ts_regret, ts_cumulative_reward = run_experiment(thompson_sampling, n_trials)


Visualization.plot_cumulative_rewards(eg_rewards, ts_rewards)
Visualization.plot_cumulative_regret(eg_regret, ts_regret)



with open("bandit_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "Trial", "Bandit", "Reward"])
    for i, reward in enumerate(eg_rewards):
        writer.writerow(["Epsilon-Greedy", i + 1, "Bandit", reward])
    for i, reward in enumerate(ts_rewards):
        writer.writerow(["Thompson Sampling", i + 1, "Bandit", reward])




print(f"Epsilon-Greedy Cumulative Reward: {eg_cumulative_reward}")
print(f"Thompson Sampling Cumulative Reward: {ts_cumulative_reward}")
