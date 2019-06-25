import gym
import numpy as np
#idk why we should reset but if you dont youll get  this 
#AssertionError: Cannot call env.step() before calling reset()

env = gym.make("MountainCar-v0")
#env.reset()


learning_rate = 0.1
discount = 0.95
episodes = 25000


#each states features are gonna be descreted
descrete_os_size = [20] * len(env.observation_space.high)
descrete_win_size = (env.observation_space.high - env.observation_space.low)/descrete_os_size

#now we have our random q_table
q_table = np.random.uniform(low=-2 , high = 0 , size = (descrete_os_size + [env.action_space.n]))

#this will discrete our state
def get_discrete_state(state):
    descrete_state = (state - env.observation_space.low)/descrete_win_size
    return tuple(descrete_state.astype(np.int))



for episode in range(episodes):
    current_d_state = get_discrete_state( env.reset())
    done = False 
    render = False
    if (episode % 500 == 0):
        render = True

    while not done:
        action = np.argmax(q_table[current_d_state])
        new_state,reward,done , _ = env.step(action)
        new_d_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            next_max_q = np.max(q_table[new_d_state]) 
            current_q = q_table[current_d_state+(action,)]
            new_current_q = (1-learning_rate) * current_q +learning_rate  * (reward + discount * next_max_q) 
            q_table[current_d_state + (action,)] = new_current_q
        elif new_state[0] >= env.goal_position:
            print(f"got it in {episode}")
            q_table[current_d_state + (action,)] = 0

        current_d_state = new_d_state
env.close()
