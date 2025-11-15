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
    Plot predator and prey trajectories with unique colors per agent.
    Works for any number of predators or prey.

    Args:
        position_history (dict):
            {
                "predators": {id: [(x0, y0), ...]},
                "prey":      {id: [(x0, y0), ...]}
            }
        grid_size (tuple): (width, height)
        save_path (str): where to save the figure
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    width, height = grid_size
    plt.figure(figsize=(6, 6))

    # Combine all agent IDs for unique color assignment
    predator_ids = list(position_history["predators"].keys())
    prey_ids = list(position_history["prey"].keys())
    total_agents = len(predator_ids) + len(prey_ids)

    # Use a categorical colormap (distinct colors)
    cmap = cm.get_cmap("tab20", total_agents)  # tab20 gives 20 distinct colors

    color_map = {}

    # Assign distinct colors to each predator
    for i, pid in enumerate(predator_ids):
        color_map[("pred", pid)] = cmap(i / total_agents)

    # Assign distinct colors to each prey (continue color indexing)
    for j, rid in enumerate(prey_ids, start=len(predator_ids)):
        color_map[("prey", rid)] = cmap(j / total_agents)

    # Plot predator trajectories
    for pid, coords in position_history["predators"].items():
        xs, ys = zip(*coords)
        color = color_map[("pred", pid)]
        plt.plot(xs, ys, "-o", color=color, label=f"Predator {pid}")
        plt.text(xs[-1], ys[-1], f"P{pid}", color=color, fontsize=9, weight="bold")

    # Plot prey trajectories
    for rid, coords in position_history["prey"].items():
        xs, ys = zip(*coords)
        color = color_map[("prey", rid)]
        plt.plot(xs, ys, "-o", color=color, label=f"Prey {rid}")
        plt.text(xs[-1], ys[-1], f"R{rid}", color=color, fontsize=9, weight="bold")

    # Format grid
    plt.xlim(-0.5, width - 0.5)
    plt.ylim(height - 0.5, -0.5)
    plt.grid(True, color="black", linewidth=0.5)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Predator–Prey Trajectories (Distinct Colors)")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.legend(loc="upper right", fontsize=8, ncol=1)
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