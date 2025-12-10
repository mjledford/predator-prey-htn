import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from constants import FIG_DIR


def record_positions(env, position_history, init=False):
    """
    Extract predator and prey positions from the POSGGym Predator-Prey env state.

    Args:
        env: POSGGym environment (with .unwrapped.state as a 3-tuple)
        position_history (dict): {"predators": {...}, "prey": {...}}
        init (bool): True to initialize new lists; False to append to existing ones
    """
    state = env.unwrapped.state
    preds = tuple(state[0])
    preys = tuple(state[1])

    if init:
        # Initialize with the starting positions
        for i, pos in enumerate(preds):
            position_history["predators"][i] = [tuple(pos)]
        for i, pos in enumerate(preys):
            position_history["prey"][i] = [tuple(pos)]
    else:
        # Append positions to existing lists
        for i, pos in enumerate(preds):
            position_history["predators"][i].append(tuple(pos))
        for i, pos in enumerate(preys):
            position_history["prey"][i].append(tuple(pos))


def plot_trajectories(position_history, grid_size=(10, 10), save_path="figures/final_positions.png"):
    """
    Plot predator and prey trajectories with fixed colors:
        Predator 0 -> green
        Predator 1 -> blue
        Prey -> red

    Final positions are highlighted with large markers.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    width, height = grid_size
    plt.figure(figsize=(6, 6))

    predator_ids = list(position_history["predators"].keys())
    prey_ids = list(position_history["prey"].keys())

    # ----- Explicit color assignment -----
    predator_colors = ["green", "blue"]   # supports 2 predators
    prey_color = "red"

    # Fallback for unusual counts
    if len(predator_ids) > len(predator_colors):
        predator_colors = predator_colors + ["cyan", "purple", "orange"][:len(predator_ids)-2]

    # ----- Plot predators -----
    for idx, pid in enumerate(predator_ids):
        coords = position_history["predators"][pid]
        xs, ys = zip(*coords)
        color = predator_colors[idx]

        # Trajectory line
        plt.plot(xs, ys, "-", color=color, label=f"Predator {pid}")

        # Normal markers for each step
        plt.plot(xs, ys, "o", color=color, markersize=4)

        # Highlight final position
        plt.plot(xs[-1], ys[-1], marker="*", color=color, markersize=14, markeredgecolor="black")

        # Label near final position
        plt.text(xs[-1] + 0.1, ys[-1] - 0.1, f"P{pid}", color=color, fontsize=9, weight="bold")

    # ----- Plot prey -----
    for rid in prey_ids:
        coords = position_history["prey"][rid]
        xs, ys = zip(*coords)
        color = prey_color

        # Trajectory line
        plt.plot(xs, ys, "-", color=color, label=f"Prey {rid}")

        # Normal markers
        plt.plot(xs, ys, "o", color=color, markersize=4)

        # Highlight final position
        plt.plot(xs[-1], ys[-1], marker="X", color=color, markersize=14, markeredgecolor="black")

        # Label near final position
        plt.text(xs[-1] + 0.1, ys[-1] - 0.1, f"R{rid}", color=color, fontsize=9, weight="bold")

    # ----- Format grid -----
    plt.xlim(-0.5, width - 0.5)
    plt.ylim(height - 0.5, -0.5)
    plt.grid(True, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Predator–Prey Trajectories")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_capture_statistics(
    all_times,
    capture_times,
    avg_capture_time,
    avg_steps_all,
    save_dir=FIG_DIR,
):
    """
    Create plots summarizing capture performance across runs.

    Parameters
    ----------
    all_times : list[int]
        Steps until capture or time_horizon for every run.
    capture_times : list[int]
        Steps until capture for successful runs only.
    avg_capture_time : float or None
        Mean capture time (successful episodes only).
    avg_steps_all : float
        Mean steps across ALL runs.
    save_dir : str
        Directory where figures will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_runs = len(all_times)
    x = list(range(1, num_runs + 1))

    # --- Plot 1: per-run capture times ---
    plt.figure()
    plt.plot(x, all_times, marker="o", linestyle="-", label="Per-run steps")

    if avg_capture_time is not None:
        plt.axhline(
            y=avg_capture_time,
            linestyle="--",
            color="red",
            label=f"Mean (captures only) = {avg_capture_time:.1f}",
        )

    plt.axhline(
        y=avg_steps_all,
        linestyle=":",
        color="green",
        label=f"Mean (all episodes) = {avg_steps_all:.1f}",
    )

    plt.xlabel("Run index")
    plt.ylabel("Steps until capture / horizon")
    plt.title("Predator-Prey: Time to Capture over Multiple Runs")
    plt.grid(True)
    plt.legend()

    out1 = os.path.join(save_dir, "time_to_capture_per_run.png")
    plt.savefig(out1, bbox_inches="tight")
    print(f"[INFO] Saved: {out1}")

    # --- Plot 2: histogram of capture times (successful only) ---
    if capture_times:
        plt.figure()
        plt.hist(capture_times, bins=20, color="steelblue", edgecolor="black")
        plt.xlabel("Steps to capture")
        plt.ylabel("Frequency")
        plt.title("Distribution of Capture Times (Successful Episodes)")
        plt.grid(True)

        out2 = os.path.join(save_dir, "capture_time_hist.png")
        plt.savefig(out2, bbox_inches="tight")
        print(f"[INFO] Saved: {out2}")
    
def plot_avg_steps_for_k(avg_capture_time, k_sync, save_dir="figs/avg_steps_k_sync"):
    """
    Saves a simple bar plot showing average steps to capture
    for a single communication frequency (k_sync).
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.title(f"Avg Steps to Capture (k_sync={k_sync})")
    plt.ylim(0, 200)
    plt.bar([1], [avg_capture_time if avg_capture_time is not None else 0], color="skyblue")
    plt.ylabel("Avg Steps to Capture")
    plt.xticks([1], [f"k={k_sync}"])
    plt.grid(axis="y")

    out_path = os.path.join(save_dir, f"avg_steps_k{k_sync}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved single-k plot to {out_path}")

def plot_k_vs_steps(results, save_path="figs/k_vs_steps.png", line=False):
    """
    Plot k_sync vs avg steps to capture.
    
    Args:
        results: dict {k: {"avg_steps": float, ...}}
        save_path: where to save the plot
        line: if True, use line plot instead of bar
    """
    ks = sorted(results.keys())
    ys = [(results[k]["avg_steps"] if results[k]["avg_steps"] is not None else 0) for k in ks]

    plt.figure(figsize=(6, 4), dpi=300)  # Set size + resolution for LaTeX
    if line:
        plt.plot(ks, ys, marker="o", color="steelblue", linewidth=2)
        for i, y in enumerate(ys):
            plt.text(ks[i], y + 4, f"{y:.1f}", ha='center', fontsize=8)
    else:
        plt.bar(ks, ys, color="steelblue", width=4)
        for i, y in enumerate(ys):
            plt.text(ks[i], y + 2, f"{y:.1f}", ha='center', fontsize=8)

    plt.xlabel("k_sync (communication interval)")
    plt.ylabel("Avg Steps to Capture")
    plt.title("Communication Frequency vs Capture Efficiency")
    plt.ylim(0, 200)
    plt.xticks(ks)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot to {save_path}")

def plot_comm_modes_comparison(results, save_path="figs/comm_modes_vs_steps.png"):
    """
    Plot comparison of communication modes.

    Args:
        results: dict {
            mode: {
                "avg_steps": float,
                "success_rate": float,
                "avg_messages": float,
                "avg_replans": float
            }
        }
    """
    modes = ["full", "periodic", "event", "none"]
    labels = ["Full", "Periodic", "Event", "None"]
    colors = ["seagreen", "dodgerblue", "orange", "crimson"]

    # Extract data
    avg_steps = [results[mode]["avg_steps"] if results[mode]["avg_steps"] is not None else 0 for mode in modes]
    success_rates = [results[mode]["success_rate"] for mode in modes]

    # Plot avg steps to capture
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, avg_steps, color=colors)
    for bar, val in zip(bars, avg_steps):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.1f}", ha="center", fontsize=8)

    plt.xlabel("Communication Mode")
    plt.ylabel("Avg Steps to Capture")
    plt.title("Effect of Communication Mode on Capture Efficiency")
    plt.ylim(0, 200)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved communication mode comparison plot to {save_path}")
    
def plot_comm_modes_success_rates(results, save_path="figs/comm_modes_vs_success.png"):
    modes = ["full", "periodic", "event", "none"]
    labels = ["Full", "Periodic", "Event", "None"]
    colors = ["seagreen", "dodgerblue", "orange", "crimson"]

    success_rates = [results[mode]["success_rate"] for mode in modes]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, success_rates, color=colors)
    for bar, val in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", fontsize=8)

    plt.xlabel("Communication Mode")
    plt.ylabel("Success Rate")
    plt.title("Effect of Communication Mode on Capture Success")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved success rate plot to {save_path}")

def plot_k_vs_costs(results, save_path_prefix="figs/k_vs"):
    """
    Plot two separate bar plots:
    - Avg Replans per Episode vs k_sync
    - Avg Messages per Episode vs k_sync

    Args:
        results: dict {k: {"avg_steps", "avg_replans", "avg_messages", ...}}
        save_path_prefix: base filepath prefix (default = "figs/k_vs")
    """
    os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)

    ks = sorted(results.keys())
    replans = [results[k]["avg_replans"] if results[k]["avg_replans"] is not None else 0 for k in ks]
    messages = [results[k]["avg_messages"] if results[k]["avg_messages"] is not None else 0 for k in ks]

    # --- Plot replans ---
    plt.figure(figsize=(5.5, 4))
    plt.bar(ks, replans, color="darkslateblue")
    for i, val in enumerate(replans):
        plt.text(ks[i], val + 1, f"{val:.1f}", ha='center', fontsize=8)
    plt.xlabel("k_sync (communication interval)")
    plt.ylabel("Avg Replans per Episode")
    plt.title("Planner Usage vs Communication Frequency")
    plt.ylim(0, max(replans) * 1.2)
    plt.grid(axis="y")
    plt.tight_layout()
    path_replans = f"{save_path_prefix}_replans.png"
    plt.savefig(path_replans, bbox_inches="tight")
    print(f"[INFO] Saved replans plot to {path_replans}")
    plt.close()

    # --- Plot messages ---
    plt.figure(figsize=(5.5, 4))
    plt.bar(ks, messages, color="seagreen")
    for i, val in enumerate(messages):
        plt.text(ks[i], val + 1, f"{val:.1f}", ha='center', fontsize=8)
    plt.xlabel("k_sync (communication interval)")
    plt.ylabel("Avg Messages per Episode")
    plt.title("Communication Cost vs Frequency")
    plt.ylim(0, max(messages) * 1.2)
    plt.grid(axis="y")
    plt.tight_layout()
    path_msgs = f"{save_path_prefix}_messages.png"
    plt.savefig(path_msgs, bbox_inches="tight")
    print(f"[INFO] Saved messages plot to {path_msgs}")
    plt.close()

