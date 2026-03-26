import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_dir(path="results"):
    os.makedirs(path, exist_ok=True)


def plot_wealth_paths(output_dir="results"):
    data = np.load("wealth_paths.npz")

    strategy_order = ["RL_gamma_1", "RL_gamma_3", "RL_gamma_8", "Baseline_60_40"]

    plt.figure(figsize=(10, 6))

    for strategy in strategy_order:
        paths = data[strategy]
        mean_path = paths.mean(axis=0)
        std_path = paths.std(axis=0)

        x = np.arange(len(mean_path))
        plt.plot(x, mean_path, label=strategy, linewidth=2)
        plt.fill_between(x, mean_path - std_path, mean_path + std_path, alpha=0.2)

    plt.title("Mean Wealth Paths with Uncertainty")
    plt.xlabel("Months")
    plt.ylabel("Wealth")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wealth_paths.png"), dpi=300)
    plt.close()


def plot_terminal_distribution(output_dir="results"):
    data = np.load("terminal_wealths.npz")

    strategy_order = ["RL_gamma_1", "RL_gamma_3", "RL_gamma_8", "Baseline_60_40"]
    terminal_values = [data[strategy] for strategy in strategy_order]

    plt.figure(figsize=(10, 6))
    plt.boxplot(terminal_values, labels=strategy_order, showfliers=False)

    plt.title("Terminal Wealth Distribution")
    plt.xlabel("Strategy")
    plt.ylabel("Terminal Wealth")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ticklabel_format(style="plain", axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "terminal_distribution.png"), dpi=300)
    plt.close()


def plot_allocation_paths(output_dir="results"):
    data = np.load("action_paths.npz")

    action_to_weight = {
        0: 0.2,
        1: 0.5,
        2: 0.8,
    }

    strategy_order = ["RL_gamma_1", "RL_gamma_3", "RL_gamma_8"]

    plt.figure(figsize=(10, 6))

    for strategy in strategy_order:
        actions = data[strategy]
        weight_paths = np.vectorize(action_to_weight.get)(actions)
        mean_weight_path = weight_paths.mean(axis=0)

        x = np.arange(len(mean_weight_path))
        plt.plot(x, mean_weight_path, label=strategy, linewidth=2)

    plt.title("Mean Stock Allocation Paths")
    plt.xlabel("Months")
    plt.ylabel("Stock Weight")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "allocation_paths.png"), dpi=300)
    plt.close()


def print_summary():
    results_df = pd.read_csv("evaluation_results.csv", index_col=0)
    print(results_df.round(4))


if __name__ == "__main__":
    ensure_output_dir()
    print_summary()
    plot_wealth_paths()
    plot_terminal_distribution()
    plot_allocation_paths()