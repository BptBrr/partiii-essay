import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
import q_network as qn
import tools
import matplotlib.pyplot as plt

# We want the agent to stop playing after the 200th step (end of the game).
terminal = 200
# We punish the games finishing before terminal time with a reward corresponding
# to the opposite of the score we want the agent to obtain.
punition = -200

# epochs : Number of times we run the SGD.

def train(agent, env, episodes, epochs):
    history = []
    for i in range(episodes):
        if (i+1) % 100 == 0:
            print "Episode %d" % (i + 1)
            #if i > 1: # We don't want to decrease at the first step
                # We decrease of 10% after 100 episodes, and stop at epsilon = 0.5%
                #agent.decrease_epsilon(0.90, 0.005)
        cur_state = env.reset()
        t = 0
        while t < terminal:
            #env.render()
            t += 1
            action = agent.get_greedy_action(cur_state)
            next_state, reward, done, info = env.step(action)
            if done and (t < terminal):
                print "Episode finished after %d timesteps" % t
                history.append(t)
                agent.append_memory([cur_state, action, next_state, -100, 1])
                agent.learn(epochs)
                break
            if done or (t == terminal):
                print "Episode succeeded after %d timesteps" % t
                history.append(t)
                agent.append_memory([cur_state, action, next_state, 1, 1])
                agent.learn(epochs)
                break
            agent.append_memory([cur_state, action, next_state, 0, done])
            cur_state = next_state
            agent.learn(epochs)

    return agent, history

# Testing function, to see if the agent is able to make up for the random
# initialization, and then succeed the episode.
def test(env, agent, init, episodes):
    history = []
    failures = 0
    for i in range(episodes):
        cur_state = env.reset()
        t = 0
        # We initalize the environment in a random setting, and check thereafter
        # if the agent we trained before is able to solve it nonetheless.
        fail = False
        while t < init:
            #env.render()
            next_state, reward, fail, info = env.step(env.action_space.sample())
            t += 1
            if fail:
                failures += 1
                print "Initialisation failure after %d steps" % t
                break
            cur_state = next_state
        if not(fail):
            while t < terminal:
                env.render()
                t += 1
                action = agent.get_best_action(cur_state)
                next_state, reward, done, info = env.step(action)
                if done or (t == terminal):
                    print "Testing episode finished after %d steps" % t
                    history.append(t)
                    break
                cur_state = next_state
    print "\n --- Summary ---"
    print "Testing procedure using %d random initialisation steps." % init
    print "Number of failures:", failures
    print "The average testing reward, calculated over %d episodes, is %d \n" %(episodes - failures,(sum(history)/len(history)))


env = gym.make('CartPole-v0')
#env = wrappers.Monitor(env, '/Users/baptiste/Programming/RLearning/Cartpole-GD3', force = True)

# We take for minibatches size something corresponding roughly to 1% of the memory size
agent = qn.Agent(0.05, 0.0001, 0.99, 128, 4, 2, 16384, 32, 16)
agent.build_network()
trained_agent, history = train(agent, env, 500, 1)
tools.rmetric(10,history)

#env.close()
#gym.upload('/Users/baptiste/Programming/RLearning/Cartpole-GD3', api_key='sk_oH6439jxQ86N7cPzqpe5tA')

test(env, trained_agent, 10, 100)

# GD : Testing reward 147
# On 2000 episodes https://gym.openai.com/evaluations/eval_ey18z0ZxRDe2MSLNmyztzw
