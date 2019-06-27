import gym
import numpy as np
import random
import matplotlib.pyplot as plt
#making the enviroment 
env = gym.make("MountainCar-v0")


# determines to what extent the next Q value overrides the current Q value
learning_rate = 0.1
# determones the importance of the future rewards
discount = 0.95
# you already know this one
episodes = 15000

every_ep = 500
# the amount of randomness
# which alows us to explore the enviroment
epsilon = 0.5

start_epsilon = 1 
end_epsilon =episodes//2
epsilon_decay = epsilon / (end_epsilon - start_epsilon)

rewards = []
debug_dic = {'ep' : [] , 'avg' : [] ,'min' : [] , 'max' : [] }

#each states features are gonna be descreted
descrete_os_size = [20] * len(env.observation_space.high)
descrete_win_size = (env.observation_space.high - env.observation_space.low)/descrete_os_size

#now we have our random q_table
q_table = np.random.uniform(low=-2 , high = 0 , size = (descrete_os_size + [env.action_space.n]))

#this will discrete our state
def get_discrete_state(state):
    descrete_state = (state - env.observation_space.low)/descrete_win_size
    return tuple(descrete_state.astype(np.int))


#lets iterate through the random
for episode in range(episodes):
    current_d_state = get_discrete_state(env.reset())
    done = False 
    render = False
    rwd = 0
    # for every 500 episodes the will be one render 
    if (episode % 2000 == 0):
        render = True

    while not done:
        #this is where epsilon comes to action
        rand = random.random()
        if rand >= epsilon:
            action = np.argmax(q_table[current_d_state])
        else:
           action = random.randint(0,2)

        new_state,reward,done , _ = env.step(action)
        new_d_state = get_discrete_state(new_state)
        
        rwd += reward 

        if render:
            env.render()
        
        if not done:
            next_max_q = np.max(q_table[new_d_state]) 
            current_q = q_table[current_d_state+(action,)]
            new_current_q = (1-learning_rate) * current_q +learning_rate  * (reward + discount * next_max_q) 
            #well this one is actually the most important one
            #its going to update the q_table
            q_table[current_d_state + (action,)] = new_current_q
        elif new_state[0] >= env.goal_position:
            print(f"got it in {episode}")
            #this one means you did great so heres your 0 reward
            #which is the highest reward
            q_table[current_d_state + (action,)] = 0
        
        current_d_state = new_d_state
    
    if end_epsilon > episode >= start_epsilon:
        epsilon -= epsilon_decay

    rewards.append(rwd)

    if episode%every_ep == 0:
        avg = sum(rewards[-every_ep:]) / len(rewards[-every_ep:])
        debug_dic['avg'].append(avg)
        debug_dic['ep'].append(episode)
        debug_dic['min'].append(min(rewards[-every_ep:]))
        debug_dic['max'].append(max(rewards[-every_ep:]))

env.close()

plt.plot( debug_dic['ep'] , debug_dic['avg'] , label = "avg" )
plt.plot( debug_dic['ep'] , debug_dic['min'] , label = "min" )
plt.plot( debug_dic['ep'] , debug_dic['max'] , label = "max" )
plt.legend()
plt.show()

#sources : https://en.wikipedia.org/wiki/Q-learning#Influence_of_variables
# and : https://www.youtube.com/watch?v=Gq1Azv_B4-4

