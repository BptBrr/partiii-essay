import gym
import tensorflow as tf
import numpy as np
import random
import q_network as qn

terminal = 300

def train(agent, env, history, episodes):
    for i in range(episodes):
        if i % 20 == 0:
            print "Episode %d" % (i + 1)
        cur_state = env.reset()
        t = 0
        episode = []
        while t < terminal:
            env.render()
            t = t+1
            action = agent.get_greedy_action(cur_state)
            next_state, reward, done, info = env.step(action)
            if done:
                reward = 0
                print "Episode finished after %d timesteps" % (t+1)
                history.append(t+1)
                break
            episode.append([cur_state, action, next_state, reward, done])
            cur_state = next_state
        agent.learn(episode, 100)
    return agent, history

env = gym.make('CartPole-v0')

agent = qn.Agent(0.05, 0.01, 0.99, 32, 4, 2, 1e4, 10, 10)
agent.build_network()
history = []
agent, history = train(agent, env, history, 80)
print history
