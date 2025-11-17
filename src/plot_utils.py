import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


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
    save_dir="figures",
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
    
def plot_avg_steps_for_k(avg_capture_time, k_sync, save_dir="figures"):
    """
    Saves a simple bar plot showing average steps to capture
    for a single communication frequency (k_sync).
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.title(f"Avg Steps to Capture (k_sync={k_sync})")
    plt.bar([1], [avg_capture_time if avg_capture_time is not None else 0])
    plt.ylabel("Avg Steps to Capture")
    plt.xticks([1], [f"k={k_sync}"])
    plt.grid(axis="y")

    out_path = os.path.join(save_dir, f"avg_steps_k{k_sync}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved single-k plot to {out_path}")

def plot_k_vs_steps(results, save_path="figures/k_vs_steps.png", line=False):
    """
    Plot k_sync vs avg steps to capture.
    results: dict {k: avg_steps}
    """

    ks = sorted(results.keys())
    ys = [results[k] if results[k] is not None else 0 for k in ks]

    plt.figure()
    if line:
        plt.plot(ks, ys, marker="o")
    else:
        plt.bar(ks, ys)

    plt.xlabel("k_sync (communication interval)")
    plt.ylabel("Avg Steps to Capture (successful episodes)")
    plt.title("Communication Frequency vs Capture Efficiency")
    plt.grid(True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"[INFO] Saved plot to {save_path}")