import random

from comm_module import CommStats

def sweep_k_sync(seed, k_values, num_episodes, time_horizon, debug, keep_prev_action):
    from run_demo import run_single_episode
    results = {}
    base_seed = seed if seed is not None else random.randint(0, 10**6)

    for k in k_values:
        capture_times = []
        stats = CommStats()  # or reset after each run, depending on how you're tracking

        for ep in range(num_episodes):
            run_seed = base_seed + ep

            captured, steps, episode_stats = run_single_episode(
                run_idx=ep,
                seed=run_seed,
                time_horizon=time_horizon,
                debug=debug,
                keep_prev_action=keep_prev_action,
                render=False,
                comm_mode="periodic",
                k_sync=k,
            )

            if captured:
                capture_times.append(steps)

            # Accumulate stats
            stats.messages += episode_stats.messages
            stats.replans += episode_stats.replans

        avg_steps = sum(capture_times) / len(capture_times) if capture_times else None
        success_rate = len(capture_times) / num_episodes
        avg_replans = stats.replans / num_episodes
        avg_messages = stats.messages / num_episodes

        results[k] = {
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "avg_replans": avg_replans,
            "avg_messages": avg_messages,
        }

    return results

def sweep_comm_modes(seed, num_episodes, time_horizon, debug, keep_prev_action, k_sync=10):
    from run_demo import run_single_episode
    comm_modes = ["full", "periodic", "event", "none"]
    results = {}

    for mode in comm_modes:
        capture_times = []
        total_msgs = 0
        total_replans = 0
        successes = 0

        for i in range(num_episodes):
            run_seed = seed + i
            captured, steps, stats = run_single_episode(
                run_idx=i,
                seed=run_seed,
                time_horizon=time_horizon,
                debug=debug,
                keep_prev_action=keep_prev_action,
                render=False,
                comm_mode=mode,
                k_sync=10,
            )
            if captured:
                capture_times.append(steps)
                successes += 1
            total_msgs += stats.messages
            total_replans += stats.replans

        avg_steps = sum(capture_times) / len(capture_times) if capture_times else None
        success_rate = successes / num_episodes
        results[mode] = {
            "avg_steps": avg_steps,
            "success_rate": success_rate,
            "avg_messages": total_msgs / num_episodes,
            "avg_replans": total_replans / num_episodes,
        }

    return results
