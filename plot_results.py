import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def plot_mean_wealth_paths():
    data = np.load("wealth_paths.npz")

    plt.figure(figsize=(10, 6))

    strategy_order = ["RL_gamma_1", "RL_gamma_3", "RL_gamma_8", "Baseline_60_40"]

    for strategy in strategy_order:
        paths = data[strategy]  # shape: (n_episodes, horizon+1)
        mean_path = paths.mean(axis=0)
        std_path = paths.std(axis=0)

        x = np.arange(len(mean_path))
        plt.plot(x, mean_path, label=strategy, linewidth=2)
        plt.fill_between(x, mean_path - std_path, mean_path + std_path, alpha=0.2)

    plt.title("Mean Wealth Paths with Uncertainty", fontsize=14)
    plt.xlabel("Months", fontsize=12)
    plt.ylabel("Wealth", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.savefig("results/wealth_paths.png", dpi=300)
    plt.close()

    print("✅ 已保存 results/wealth_paths.png")


def plot_terminal_distribution():
    data = np.load("terminal_wealths.npz")

    strategy_order = ["RL_gamma_1", "RL_gamma_3", "RL_gamma_8", "Baseline_60_40"]
    terminal_data = [data[strategy] for strategy in strategy_order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(terminal_data, labels=strategy_order, showfliers=False)

    plt.title("Terminal Wealth Distribution", fontsize=14)
    plt.xlabel("Strategy", fontsize=12)
    plt.ylabel("Terminal Wealth", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.savefig("results/terminal_distribution.png", dpi=300)
    plt.close()

    print("✅ 已保存 results/terminal_distribution.png")


def plot_allocation_paths():
    data = np.load("action_paths.npz")

    # 动作到股票权重的映射
    action_to_weight = {
        0: 0.2,
        1: 0.5,
        2: 0.8,
    }

    strategy_order = ["RL_gamma_1", "RL_gamma_3", "RL_gamma_8"]

    plt.figure(figsize=(10, 6))

    for strategy in strategy_order:
        actions = data[strategy]  # shape: (n_episodes, horizon)
        weight_paths = np.vectorize(action_to_weight.get)(actions)
        mean_weight_path = weight_paths.mean(axis=0)

        x = np.arange(len(mean_weight_path))
        plt.plot(x, mean_weight_path, label=strategy, linewidth=2)

    plt.title("Mean Stock Allocation Paths", fontsize=14)
    plt.xlabel("Months", fontsize=12)
    plt.ylabel("Stock Weight", fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/allocation_paths.png", dpi=300)
    plt.close()

    print("✅ 已保存 results/allocation_paths.png")


def print_summary_table():
    df = pd.read_csv("evaluation_results.csv", index_col=0)
    print("\n================ Evaluation Summary ================")
    print(df.round(4))


if __name__ == "__main__":
    ensure_results_dir()
    print_summary_table()
    plot_mean_wealth_paths()
    plot_terminal_distribution()
    plot_allocation_paths()