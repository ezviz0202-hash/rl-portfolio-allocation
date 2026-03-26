import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from data_loader import load_returns
from retirement_env import RetirementEnv


def create_env(risk_gamma=3.0):
    stock_returns, bond_returns = load_returns("market_data.csv")

    env = RetirementEnv(
        stock_returns=stock_returns,
        bond_returns=bond_returns,
        gamma=risk_gamma,
        initial_wealth=100000.0,
        target_wealth=500000.0,
        horizon_months=180,
    )
    return env


def run_rl_episode(model, env):
    obs, _ = env.reset()

    wealth_path = [env.wealth]
    action_path = []

    done = False
    truncated = False

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        obs, reward, done, truncated, info = env.step(action)
        wealth_path.append(info["wealth"])
        action_path.append(action)

    return wealth_path, action_path


def run_baseline_episode(env, stock_weight=0.6):
    obs, _ = env.reset()

    wealth = env.wealth
    wealth_path = [wealth]

    start_idx = env.start_idx
    horizon = env.horizon_months

    for t in range(horizon):
        idx = start_idx + t

        stock_r = float(env.stock_returns[idx])
        bond_r = float(env.bond_returns[idx])

        bond_weight = 1.0 - stock_weight
        portfolio_return = stock_weight * stock_r + bond_weight * bond_r

        wealth = wealth * (1.0 + portfolio_return)
        wealth_path.append(wealth)

    return wealth_path


def compute_max_drawdown(wealth_path):
    wealth_array = np.array(wealth_path, dtype=np.float64)
    running_max = np.maximum.accumulate(wealth_array)
    drawdowns = (running_max - wealth_array) / running_max
    return float(np.max(drawdowns))


def summarize_results(terminal_wealths, wealth_paths, target_wealth):
    terminal_wealths = np.array(terminal_wealths, dtype=np.float64)
    max_drawdowns = [compute_max_drawdown(path) for path in wealth_paths]

    summary = {
        "Mean Terminal Wealth": float(np.mean(terminal_wealths)),
        "Median Terminal Wealth": float(np.median(terminal_wealths)),
        "Std Terminal Wealth": float(np.std(terminal_wealths)),
        "Goal Hit Rate": float(np.mean(terminal_wealths >= target_wealth)),
        "Mean Max Drawdown": float(np.mean(max_drawdowns)),
    }
    return summary


def evaluate_rl_model(model_path, risk_gamma, n_episodes=100):
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

    summary = summarize_results(
        terminal_wealths=terminal_wealths,
        wealth_paths=wealth_paths,
        target_wealth=env.target_wealth
    )
    return summary, terminal_wealths, wealth_paths, action_paths


def evaluate_baseline(n_episodes=100, stock_weight=0.6):
    env = create_env(risk_gamma=3.0)

    terminal_wealths = []
    wealth_paths = []

    for _ in range(n_episodes):
        wealth_path = run_baseline_episode(env, stock_weight=stock_weight)
        wealth_paths.append(wealth_path)
        terminal_wealths.append(wealth_path[-1])

    summary = summarize_results(
        terminal_wealths=terminal_wealths,
        wealth_paths=wealth_paths,
        target_wealth=env.target_wealth
    )
    return summary, terminal_wealths, wealth_paths


if __name__ == "__main__":
    n_episodes = 100

    results = {}
    terminal_dict = {}
    path_dict = {}
    action_dict = {}

    rl_configs = [
        ("RL_gamma_1", "models/dqn_risk_gamma_1.0.zip", 1.0),
        ("RL_gamma_3", "models/dqn_risk_gamma_3.0.zip", 3.0),
        ("RL_gamma_8", "models/dqn_risk_gamma_8.0.zip", 8.0),
    ]

    for strategy_name, model_path, risk_gamma in rl_configs:
        print(f"\n🚀 正在评估 {strategy_name} ...")
        summary, terminal_wealths, wealth_paths, action_paths = evaluate_rl_model(
            model_path=model_path,
            risk_gamma=risk_gamma,
            n_episodes=n_episodes
        )
        results[strategy_name] = summary
        terminal_dict[strategy_name] = np.array(terminal_wealths, dtype=np.float64)
        path_dict[strategy_name] = np.array(wealth_paths, dtype=np.float64)
        action_dict[strategy_name] = np.array(action_paths, dtype=np.int32)

    print("\n🚀 正在评估 Baseline_60_40 ...")
    summary, terminal_wealths, wealth_paths = evaluate_baseline(
        n_episodes=n_episodes,
        stock_weight=0.6
    )
    results["Baseline_60_40"] = summary
    terminal_dict["Baseline_60_40"] = np.array(terminal_wealths, dtype=np.float64)
    path_dict["Baseline_60_40"] = np.array(wealth_paths, dtype=np.float64)

    df_results = pd.DataFrame(results).T
    print("\n================ Monte Carlo Evaluation Results ================")
    print(df_results)

    df_results.to_csv("evaluation_results.csv")
    np.savez("terminal_wealths.npz", **terminal_dict)
    np.savez("wealth_paths.npz", **path_dict)

    # 只有 RL 策略有 action paths
    np.savez("action_paths.npz", **action_dict)

    print("\n✅ 已保存：")
    print("  - evaluation_results.csv")
    print("  - terminal_wealths.npz")
    print("  - wealth_paths.npz")
    print("  - action_paths.npz")