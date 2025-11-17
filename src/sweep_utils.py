import random
from run_demo import run_single_episode

def sweep_k_sync(k_values, num_episodes, time_horizon, debug, keep_prev_action):
    """
    Minimal sweep over k_sync values.
    Returns: dict {k_sync: avg_steps_to_capture}
    """
    results = {}

    base_seed = random.randint(0, 10**6)

    for k in k_values:
        capture_times = []

        for ep in range(num_episodes):
            seed = base_seed + ep   # same seeds for all k values = fair comparison

            captured, steps = run_single_episode(
                seed=seed,
                time_horizon=time_horizon,
                debug=debug,
                keep_prev_action=keep_prev_action,
                render=False,
                comm_mode="periodic",
                k_sync=k,
            )

            if captured:
                capture_times.append(steps)

        # Compute average (None → no captures)
        avg_steps = sum(capture_times) / len(capture_times) if capture_times else None
        results[k] = avg_steps

    return results