import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from data_loader import load_returns
from retirement_env import RetirementEnv


def create_env(risk_gamma=3.0):
    stock_returns, bond_returns = load_returns("market_data.csv")

    return RetirementEnv(
        stock_returns=stock_returns,
        bond_returns=bond_returns,
        gamma=risk_gamma,
        initial_wealth=100000.0,
        target_wealth=500000.0,
        horizon_months=180,
    )


def run_rl_episode(model, env):
    obs, _ = env.reset()

    wealth_path = [env.wealth]
    action_path = []

    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, _, terminated, truncated, info = env.step(action)

        wealth_path.append(info["wealth"])
        action_path.append(action)

    return wealth_path, action_path


def run_fixed_weight_episode(env, stock_weight=0.6):
    obs, _ = env.reset()

    wealth = env.wealth
    wealth_path = [wealth]

    start_idx = env.start_idx
    horizon = env.horizon_months

    for t in range(horizon):
        idx = start_idx + t

        stock_return = float(env.stock_returns[idx])
        bond_return = float(env.bond_returns[idx])

        bond_weight = 1.0 - stock_weight
        portfolio_return = stock_weight * stock_return + bond_weight * bond_return

        wealth *= 1.0 + portfolio_return
        wealth_path.append(wealth)

    return wealth_path


def compute_max_drawdown(wealth_path):
    wealth_array = np.asarray(wealth_path, dtype=np.float64)
    running_max = np.maximum.accumulate(wealth_array)
    drawdown = (running_max - wealth_array) / running_max
    return float(np.max(drawdown))


def summarize_performance(terminal_wealths, wealth_paths, target_wealth):
    terminal_wealths = np.asarray(terminal_wealths, dtype=np.float64)
    max_drawdowns = [compute_max_drawdown(path) for path in wealth_paths]

    return {
        "Mean Terminal Wealth": float(np.mean(terminal_wealths)),
        "Median Terminal Wealth": float(np.median(terminal_wealths)),
        "Std Terminal Wealth": float(np.std(terminal_wealths)),
        "Goal Hit Rate": float(np.mean(terminal_wealths >= target_wealth)),
        "Mean Max Drawdown": float(np.mean(max_drawdowns)),
    }


def evaluate_rl_strategy(model_path, risk_gamma, n_episodes=100):
    env = create_env(risk_gamma=risk_gamma)
    model = DQN.load(model_path)

    terminal_wealths = []
    wealth_paths = []
    action_paths = []

    for _ in range(n_episodes):
        wealth_path, action_path = run_rl_episode(model, env)
        wealth_paths.append(wealth_path)
        action_paths.append(action_path)
        terminal_wealths.append(wealth_path[-1])

    summary = summarize_performance(
        terminal_wealths=terminal_wealths,
        wealth_paths=wealth_paths,
        target_wealth=env.target_wealth,
    )

    return summary, terminal_wealths, wealth_paths, action_paths


def evaluate_fixed_weight_strategy(stock_weight=0.6, n_episodes=100):
    env = create_env(risk_gamma=3.0)

    terminal_wealths = []
    wealth_paths = []

    for _ in range(n_episodes):
        wealth_path = run_fixed_weight_episode(env, stock_weight=stock_weight)
        wealth_paths.append(wealth_path)
        terminal_wealths.append(wealth_path[-1])

    summary = summarize_performance(
        terminal_wealths=terminal_wealths,
        wealth_paths=wealth_paths,
        target_wealth=env.target_wealth,
    )

    return summary, terminal_wealths, wealth_paths


if __name__ == "__main__":
    n_episodes = 100

    summary_table = {}
    terminal_wealths_dict = {}
    wealth_paths_dict = {}
    action_paths_dict = {}

    rl_configs = [
        ("RL_gamma_1", "models/dqn_risk_gamma_1.0.zip", 1.0),
        ("RL_gamma_3", "models/dqn_risk_gamma_3.0.zip", 3.0),
        ("RL_gamma_8", "models/dqn_risk_gamma_8.0.zip", 8.0),
    ]

    for strategy_name, model_path, risk_gamma in rl_configs:
        summary, terminal_wealths, wealth_paths, action_paths = evaluate_rl_strategy(
            model_path=model_path,
            risk_gamma=risk_gamma,
            n_episodes=n_episodes,
        )

        summary_table[strategy_name] = summary
        terminal_wealths_dict[strategy_name] = np.asarray(terminal_wealths, dtype=np.float64)
        wealth_paths_dict[strategy_name] = np.asarray(wealth_paths, dtype=np.float64)
        action_paths_dict[strategy_name] = np.asarray(action_paths, dtype=np.int32)

    summary, terminal_wealths, wealth_paths = evaluate_fixed_weight_strategy(
        stock_weight=0.6,
        n_episodes=n_episodes,
    )

    summary_table["Baseline_60_40"] = summary
    terminal_wealths_dict["Baseline_60_40"] = np.asarray(terminal_wealths, dtype=np.float64)
    wealth_paths_dict["Baseline_60_40"] = np.asarray(wealth_paths, dtype=np.float64)

    results_df = pd.DataFrame(summary_table).T
    print(results_df)

    results_df.to_csv("evaluation_results.csv")
    np.savez("terminal_wealths.npz", **terminal_wealths_dict)
    np.savez("wealth_paths.npz", **wealth_paths_dict)
    np.savez("action_paths.npz", **action_paths_dict)