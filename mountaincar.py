import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
import q_network as qn
import tools

terminal = 200
useGymTools = False

def train(agent, env, episodes, epochs, number_random_states, render):

    # For the average total reward metric :
    reward_history = []
    # For the maximum predicted Q value metric :
    random_states = []
    q_history = []

    if not(useGymTools):
        for i in range(number_random_states):
            cur_state = env.reset()
            for j in range(5):
                next_state, reward, fail, info = env.step(env.action_space.sample())
            random_states.append(next_state)

    for i in range(episodes):
        if (i+1) % 100 == 0:
            print "Episode %d" % (i + 1)

        cur_state = env.reset()
        t = 0

        if agent.anneal:
            agent.decrease_epsilon(0.002)

        while t < terminal:
            if render:
                env.render()
            t += 1
            action = agent.get_greedy_action(cur_state)
            next_state, reward, done, info = env.step(action)

            # We take for reward the change in game score : if the car goes up, positive reward
            #reward = np.sign(next_state[0] - cur_state[0])

            if done and (t < terminal):
                print "Episode succeeded after %d timesteps" % t
                agent.append_memory([cur_state, action, next_state, (200 - t), 1])
                agent.learn(epochs)
                reward_history.append(-t)
                if not(useGymTools):
                    q_history.append(np.mean(agent.get_max_q(random_states)))
                break

            if done or (t == terminal):
                print "Episode failed after %d timesteps" % t
                agent.append_memory([cur_state, action, next_state, -200, 1])
                agent.learn(epochs)
                reward_history.append(-t)
                if not(useGymTools):
                    q_history.append(np.mean(agent.get_max_q(random_states)))
                break

            agent.append_memory([cur_state, action, next_state, 0, done])
            cur_state = next_state
            agent.learn(epochs)
    return agent, reward_history, q_history

env = gym.make('MountainCar-v0')
action_size = env.action_space.n
state_size = env.observation_space.high.shape[0]

if useGymTools:
    env = wrappers.Monitor(env, '/Users/baptiste/Programming/RLearning/MountainCar-RMSProp1', force = True)

agent = qn.Agent(1, 0.00019, 0.99, 256, state_size, action_size, 1048576, 128, 64, True, 0.01)
agent.build_network()
agent, r_history, q_history = train(agent, env, 5000, 1, 32, False)
print r_history
tools.rmetric(10, r_history)
tools.qmetric(q_history)

if useGymTools:
    env.close()
    gym.upload('/Users/baptiste/Programming/RLearning/MountainCar-RMSProp1', api_key='sk_oH6439jxQ86N7cPzqpe5tA')
