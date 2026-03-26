import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RetirementEnv(gym.Env):
    """
    生命周期资产配置环境（版本二稳定版，无 drawdown）
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        stock_returns,
        bond_returns,
        gamma=3.0,
        initial_wealth=100000.0,
        target_wealth=500000.0,
        horizon_months=180,
    ):
        super(RetirementEnv, self).__init__()

        self.stock_returns = np.asarray(stock_returns, dtype=np.float32)
        self.bond_returns = np.asarray(bond_returns, dtype=np.float32)

        if len(self.stock_returns) != len(self.bond_returns):
            raise ValueError("stock_returns 和 bond_returns 长度必须一致。")

        if len(self.stock_returns) < horizon_months:
            raise ValueError(
                f"历史数据长度不足。当前长度={len(self.stock_returns)}，"
                f"但 horizon_months={horizon_months}"
            )

        self.gamma = float(gamma)
        self.initial_wealth = float(initial_wealth)
        self.target_wealth = float(target_wealth)
        self.horizon_months = int(horizon_months)

        # 动作空间（3种资产配置）
        self.action_space = spaces.Discrete(3)
        self.action_to_stock_weight = {
            0: 0.2,
            1: 0.5,
            2: 0.8,
        }

        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        # 内部变量
        self.start_idx = None
        self.current_step = None
        self.wealth = None
        self.last_stock_return = None
        self.last_bond_return = None

    def utility(self, wealth):
        w = max(float(wealth), 1e-6)

        if self.gamma == 1.0:
            return np.log(w)
        return (w ** (1.0 - self.gamma)) / (1.0 - self.gamma)

    def _get_obs(self):
        remaining_months = self.horizon_months - self.current_step
        funding_ratio = self.wealth / self.target_wealth

        obs = np.array(
            [
                self.wealth,
                remaining_months,
                self.last_stock_return,
                self.last_bond_return,
                funding_ratio,
            ],
            dtype=np.float32,
        )
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.stock_returns) - self.horizon_months
        self.start_idx = self.np_random.integers(0, max_start + 1)

        self.current_step = 0
        self.wealth = self.initial_wealth

        self.last_stock_return = float(self.stock_returns[self.start_idx])
        self.last_bond_return = float(self.bond_returns[self.start_idx])

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        if action not in self.action_to_stock_weight:
            raise ValueError(f"非法动作: {action}")

        idx = self.start_idx + self.current_step

        stock_r = float(self.stock_returns[idx])
        bond_r = float(self.bond_returns[idx])

        stock_weight = self.action_to_stock_weight[int(action)]
        bond_weight = 1.0 - stock_weight

        portfolio_return = stock_weight * stock_r + bond_weight * bond_r

        prev_wealth = self.wealth
        self.wealth = self.wealth * (1.0 + portfolio_return)

        # ⭐ 核心 reward（版本二）
        reward = (self.utility(self.wealth) - self.utility(prev_wealth)) * 1e12

        self.last_stock_return = stock_r
        self.last_bond_return = bond_r
        self.current_step += 1

        terminated = self.current_step >= self.horizon_months
        truncated = False

        obs = self._get_obs()
        info = {
            "wealth": self.wealth,
            "stock_return": stock_r,
            "bond_return": bond_r,
            "stock_weight": stock_weight,
            "bond_weight": bond_weight,
            "portfolio_return": portfolio_return,
        }

        return obs, float(reward), terminated, truncated, info


if __name__ == "__main__":
    from data_loader import load_returns

    stock_returns, bond_returns = load_returns("market_data.csv")

    env = RetirementEnv(
        stock_returns=stock_returns,
        bond_returns=bond_returns,
        gamma=3.0,
        initial_wealth=100000.0,
        target_wealth=500000.0,
        horizon_months=180,
    )

    obs, info = env.reset()
    print("✅ 环境初始化成功")
    print("初始状态:", obs)

    action_names = {
        0: "20% stock / 80% bond",
        1: "50% stock / 50% bond",
        2: "80% stock / 20% bond",
    }

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\n第 {i+1} 步")
        print("动作:", action, action_names[action])
        print("状态:", obs)
        print("reward:", reward)
        print("财富:", info["wealth"])
        print("组合收益率:", info["portfolio_return"])

        if terminated:
            print("Episode 结束")
            break