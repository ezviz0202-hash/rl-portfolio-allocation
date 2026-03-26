import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RetirementEnv(gym.Env):
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
        super().__init__()

        self.stock_returns = np.asarray(stock_returns, dtype=np.float32)
        self.bond_returns = np.asarray(bond_returns, dtype=np.float32)

        if len(self.stock_returns) != len(self.bond_returns):
            raise ValueError("stock_returns and bond_returns must have the same length.")

        if len(self.stock_returns) < horizon_months:
            raise ValueError(
                f"Insufficient return history: {len(self.stock_returns)} < {horizon_months}"
            )

        self.gamma = float(gamma)
        self.initial_wealth = float(initial_wealth)
        self.target_wealth = float(target_wealth)
        self.horizon_months = int(horizon_months)

        self.action_space = spaces.Discrete(3)
        self.stock_weights = {
            0: 0.2,
            1: 0.5,
            2: 0.8,
        }

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, 1.0, 1.0, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        self.start_idx = None
        self.current_step = None
        self.wealth = None
        self.last_stock_return = None
        self.last_bond_return = None

    def utility(self, wealth):
        wealth = max(float(wealth), 1e-6)

        if self.gamma == 1.0:
            return np.log(wealth)
        return (wealth ** (1.0 - self.gamma)) / (1.0 - self.gamma)

    def _get_obs(self):
        remaining_months = self.horizon_months - self.current_step
        funding_ratio = self.wealth / self.target_wealth

        return np.array(
            [
                self.wealth,
                remaining_months,
                self.last_stock_return,
                self.last_bond_return,
                funding_ratio,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.stock_returns) - self.horizon_months
        self.start_idx = self.np_random.integers(0, max_start + 1)

        self.current_step = 0
        self.wealth = self.initial_wealth
        self.last_stock_return = float(self.stock_returns[self.start_idx])
        self.last_bond_return = float(self.bond_returns[self.start_idx])

        return self._get_obs(), {}

    def step(self, action):
        action = int(action)

        if action not in self.stock_weights:
            raise ValueError(f"Invalid action: {action}")

        idx = self.start_idx + self.current_step
        stock_return = float(self.stock_returns[idx])
        bond_return = float(self.bond_returns[idx])

        stock_weight = self.stock_weights[action]
        bond_weight = 1.0 - stock_weight
        portfolio_return = stock_weight * stock_return + bond_weight * bond_return

        prev_wealth = self.wealth
        self.wealth *= 1.0 + portfolio_return

        reward = (self.utility(self.wealth) - self.utility(prev_wealth)) * 1e12

        self.last_stock_return = stock_return
        self.last_bond_return = bond_return
        self.current_step += 1

        terminated = self.current_step >= self.horizon_months
        truncated = False

        info = {
            "wealth": self.wealth,
            "stock_return": stock_return,
            "bond_return": bond_return,
            "stock_weight": stock_weight,
            "bond_weight": bond_weight,
            "portfolio_return": portfolio_return,
        }

        return self._get_obs(), float(reward), terminated, truncated, info