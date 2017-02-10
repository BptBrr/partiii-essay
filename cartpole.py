import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
import q_network as qn

# We want the agent to stop playing after the 200th step (end of the game).
terminal = 200
# We punish the games finishing before terminal time with a reward corresponding
# to the opposite of the score we want the agent to obtain.
punition = -200

def train(agent, env, history, episodes):
    for i in range(episodes):
        if (i+1) % 50 == 0:
            print "Episode %d" % (i + 1)
        cur_state = env.reset()
        t = 0
        episode = []
        while t < terminal:
            #env.render()
            t += 1
            action = agent.get_greedy_action(cur_state)
            next_state, reward, done, info = env.step(action)
            if done and (t < terminal):
                print "Episode finished after %d timesteps" % t
                history.append(t)
                episode.append([cur_state, action, next_state, punition, 1])
                agent.learn(episode,1)
                break
            if done or (t == terminal):
                print "Episode succeeded after %d timesteps" % t
                history.append(t)
                episode.append([cur_state, action, next_state, reward, 1])
                agent.learn(episode,1)
                break
            episode.append([cur_state, action, next_state, reward, done])
            cur_state = next_state
            agent.learn(episode,1)
    return agent, history

env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, '/Users/baptiste/Programming/RLearning/Cartpole', force = True)

# We take for minibatches size something corresponding roughly to 1% of the memory size
agent = qn.Agent(0.05, 0.0002, 0.99, 128, 4, 2, 10000, 20, 20)
agent.build_network()
history = []
agent, history = train(agent, env, history, 2000)
print history

#env.close()
#gym.upload('/Users/baptiste/Programming/RLearning/Cartpole', api_key='sk_oH6439jxQ86N7cPzqpe5tA')
