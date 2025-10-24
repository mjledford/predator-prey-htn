
import time
import inspect
from pprint import pprint

import posggym
# actions: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT (per this env)
# obs cells: 0=EMPTY, 1=WALL, 2=PREDATOR, 3=PREY

import posggym.envs.grid_world.predator_prey as pp

from wrappers import ActionLoggingWrapper
from observers import HelloObserver

import gtpyhop
from pp_htn import domain_name


#print(pp.__file__)

ACTION_NAMES = {
    0: "STAY",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

def plan_to_actions(plan):
    """Plan should be in the list[tuple] for: [('do', 0, 4)] --> agent 0 do action 4"""
    for op, aid, act in plan:
        return act
    

def my_policy(obs, agent_id):
    """A very simple policy for testing purposes."""
    # Move up
    return 1

def main():
    #print("Testing environment creation...")
    time_horizon=10
    TARGET_FPS = 5
    SLEEP = 1.0 / TARGET_FPS
    
    env = posggym.make(
        "PredatorPrey-v0",
        max_episode_steps=200,
        grid="10x10",
        num_predators=2,
        num_prey=1,
        render_mode="human",
    )
    env = ActionLoggingWrapper(env, debug=True)
   
    print(f"[INFO] Printing GTPyhop Domain")
    gtpyhop.print_domain()
    print("======================================")
    observations, infos = env.reset(seed=42)
    print(f"Starting episode with agents: {env.agents}")
    print("env.agents:", list(env.agents))
    print("obs keys:  ", list(observations.keys()))
    for aid in env.agents:
        print(aid, "action space:", env.action_spaces[aid])
    
    observer = HelloObserver(pretty=False)
    observer.on_reset(env, observations,infos)
    
    
    

    for t in range(time_horizon):
        # Build a GTPyhop state with exactly the methods we need
        s = gtpyhop.State("tick")
        s.obs_dim = env.unwrapped.model.obs_dim
        s.obs = {agent_id: observations[agent_id] for agent_id in env.agents }
        
        # PLan one primitive action per agent
        actions = {}
        for agent_id in env.agents:
            if not hasattr(s, "last_action"):
                s.last_action = {}
            else:
                s.last_action.clear()
            plan = gtpyhop.find_plan(s, [("choose_action", agent_id)])
            action = plan_to_actions(plan)
            #print(f"Plan: {type(plan)} for Agent {agent_id}")
            # example: plan[('do', '0', '4')]
            #act = s.last_action.get(agent_id,0) if plan else 0 #fallback STAY
            actions[agent_id] = int(action)
            
        print(f"Actions type {type(actions)}")
        readable = {aid: f"{act} ({ACTION_NAMES[act]})" for aid, act in actions.items()}
        pprint(readable)
        print(f"Actions type: {type(actions)}")
        print("=======================")
        print(actions)
        print("=========================")
        # 2) keep actions random for now (hello world)
        #actions = {agent_id: env.action_spaces[agent_id].sample() for agent_id in env.agents}

        # step environment
        observations, rewards, terminations, truncations, all_done, infos = env.step(actions)
        observer.on_step(t, observations, rewards, terminations, truncations, infos)
        # 4) compact tick summary
        pprint(
            f"[t={t}] | obs={observations} "
            f"| rewards={len(rewards)} | done={all_done} "
            f"| term={sum(1 for v in terminations.values() if v)} "
            f"| trunc={sum(1 for v in truncations.values() if v)}"
        )

        env.render()
        #time.sleep(0.03)
        time.sleep(SLEEP)

        if all_done:
            reason = "time_limit" if any(truncations.values()) else "task_solved"
            #print(f"Episode ended at t={t} due to {reason}")
            observer.on_episode_end(reason)
            break

    print(f"Episode finished after {t} steps")
    env.close()


if __name__ == "__main__":
    main()

# def reshape5(obs):
#     return [obs[i:i+5] for i in range(0, 25, 5)]

# def prey_offsets(obs):
#     PREY = 3
#     out = []
#     for k, v in enumerate(obs):
#         if v == PREY:
#             row, col = divmod(k, 5)
#             out.append((col-2, row-2))  # (dx, dy)
#     return out

# for aid, obs in observations.items():
#     print(f"\nagent {aid}")
#     for row in reshape5(obs):
#         print(" ".join(map(str,row)))
#     print("prey rel offsets (dx,dy):", prey_offsets(obs))
    