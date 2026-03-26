import os
from collections import Counter

from stable_baselines3 import DQN

from data_loader import load_returns
from retirement_env import RetirementEnv


def create_env(risk_gamma):
    stock_returns, bond_returns = load_returns("market_data.csv")

    return RetirementEnv(
        stock_returns=stock_returns,
        bond_returns=bond_returns,
        gamma=risk_gamma,
        initial_wealth=100000.0,
        target_wealth=500000.0,
        horizon_months=180,
    )


def train_model(risk_gamma, total_timesteps=10000):
    env = create_env(risk_gamma)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
    )

    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/dqn_risk_gamma_{risk_gamma}.zip"
    model.save(model_path)

    return model, env, model_path


def run_episode(model, env):
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


def action_distribution(action_path):
    counts = Counter(action_path)
    total = len(action_path)

    distribution = {}
    for action in [0, 1, 2]:
        distribution[action] = {
            "count": counts.get(action, 0),
            "ratio": counts.get(action, 0) / total if total > 0 else 0.0,
        }
    return distribution


if __name__ == "__main__":
    risk_gammas = [1.0, 3.0, 8.0]
    results = {}

    for risk_gamma in risk_gammas:
        model, env, model_path = train_model(
            risk_gamma=risk_gamma,
            total_timesteps=10000,
        )

        wealth_path, action_path = run_episode(model, env)
        final_wealth = wealth_path[-1]
        distribution = action_distribution(action_path)

        print(f"\nrisk_gamma = {risk_gamma}")
        print(f"model_path = {model_path}")
        print(f"final_wealth = {final_wealth:.2f}")
        print(f"path_length = {len(wealth_path)}")
        print(f"first_20_actions = {action_path[:20]}")
        print(f"last_20_actions = {action_path[-20:]}")
        print("action_distribution = {")
        for action in [0, 1, 2]:
            count = distribution[action]['count']
            ratio = distribution[action]['ratio']
            print(f"  {action}: count={count}, ratio={ratio:.2%}")
        print("}")

        results[risk_gamma] = {
            "final_wealth": final_wealth,
            "model_path": model_path,
            "action_distribution": distribution,
        }