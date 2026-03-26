import os
from collections import Counter

from stable_baselines3 import DQN

from data_loader import load_returns
from retirement_env import RetirementEnv


def create_env(risk_gamma):
    stock_returns, bond_returns = load_returns("market_data.csv")

    env = RetirementEnv(
        stock_returns=stock_returns,
        bond_returns=bond_returns,
        gamma=risk_gamma,           # 这是 CRRA 风险厌恶参数
        initial_wealth=100000.0,
        target_wealth=500000.0,
        horizon_months=180,
    )
    return env


def train_single_model(risk_gamma, total_timesteps=20000):
    env = create_env(risk_gamma)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,                 # 这是 DQN 折扣因子，不是风险偏好 gamma
        train_freq=4,
        target_update_interval=500,
    )

    print(f"\n🚀 开始训练模型：risk_gamma = {risk_gamma}")
    model.learn(total_timesteps=total_timesteps)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/dqn_risk_gamma_{risk_gamma}.zip"
    model.save(save_path)
    print(f"✅ 模型已保存到: {save_path}")

    return model, env, save_path


def test_single_episode(model, env):
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


def summarize_actions(action_path):
    action_counter = Counter(action_path)
    total = len(action_path)

    action_names = {
        0: "20% stock / 80% bond",
        1: "50% stock / 50% bond",
        2: "80% stock / 20% bond",
    }

    print("动作分布：")
    for action in [0, 1, 2]:
        count = action_counter.get(action, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  动作 {action} ({action_names[action]}): {count} 次, {pct:.2f}%")


if __name__ == "__main__":
    risk_gamma_list = [1.0, 3.0, 8.0]

    all_results = {}

    for risk_gamma in risk_gamma_list:
        model, env, save_path = train_single_model(
            risk_gamma=risk_gamma,
            total_timesteps=10000
        )

        wealth_path, action_path = test_single_episode(model, env)

        final_wealth = wealth_path[-1]

        print(f"\n📊 risk_gamma = {risk_gamma} 测试完成")
        print("模型路径:", save_path)
        print("最终财富:", final_wealth)
        print("路径长度:", len(wealth_path))
        print("前20个动作:", action_path[:20])
        print("后20个动作:", action_path[-20:])
        summarize_actions(action_path)

        all_results[risk_gamma] = {
            "final_wealth": final_wealth,
            "wealth_path": wealth_path,
            "action_path": action_path,
            "model_path": save_path,
        }

    print("\n================ 最终汇总 ================")
    for risk_gamma, result in all_results.items():
        print(f"risk_gamma = {risk_gamma}")
        print(f"  最终财富: {result['final_wealth']}")
        print(f"  模型路径: {result['model_path']}")
        print("-" * 40)