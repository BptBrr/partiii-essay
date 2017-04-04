import gym
from gym import wrappers
import tensorflow as tf
import numpy as np
import random
import q_network as qn
import tools

# We want the agent to stop playing after the 200th step (end of the game).
terminal = 200

# Controls the use of gym wrapper, which sends results to the OpenAI servers
# for analysis. Since this conflicts with our custom Q-metric, we need to make
# different cases in the train function, using this boolean.
useGymTools = False

# epochs : Number of times we run the SGD.

def train(agent, env, episodes, epochs, number_random_states, render):
    # Initialize the lists that will allow us to plot assessment metrics
    # For the average total reward metric :
    reward_history = []
    # For the maximum predicted Q value metric :
    random_states = []
    q_history = []

    # We get some random states, necessary for the Q metric.
    if not(useGymTools):
        for i in range(number_random_states):
            cur_state = env.reset()
            for j in range(5):
                next_state, reward, fail, info = env.step(env.action_space.sample())
            random_states.append(next_state)

    # The learning loop :
    for i in range(episodes):
        if (i+1) % 100 == 0:
            print "Episode %d" % (i + 1)

        # At each step, we decrease epsilon.
        if agent.anneal:
            agent.decrease_epsilon(0.001)

        cur_state = env.reset()
        t = 0

        while t < terminal:
            if render:
                env.render()
            t += 1
            action = agent.get_greedy_action(cur_state)
            next_state, reward, done, info = env.step(action)
            if done and (t < terminal):
                print "Episode finished after %d timesteps" % t
                agent.append_memory([cur_state, action, next_state, 0, 1])
                agent.learn(epochs)
                reward_history.append(t)
                if not(useGymTools):
                    q_history.append(np.mean(agent.get_max_q(random_states)))

                break
            if done or (t == terminal):
                print "Episode succeeded after %d timesteps" % t
                agent.append_memory([cur_state, action, next_state, 1, 1])
                agent.learn(epochs)
                reward_history.append(t)
                if not(useGymTools):
                    q_history.append(np.mean(agent.get_max_q(random_states)))
                break
            agent.append_memory([cur_state, action, next_state, 1, done])
            cur_state = next_state
            agent.learn(epochs)

    return agent, reward_history, q_history

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
        # fail is here so that we do not count an initialisation failure as a
        # failure of the agent.
        fail = False
        while t < init:
            env.render()
            next_state, reward, fail, info = env.step(env.action_space.sample())
            t += 1
            if fail:
                failures += 1
                print "Initialisation failure after %d steps" % t
                break
            cur_state = next_state
        if not(fail):
            # After initialisation, we finish the episode using the trained agent.
            # We do not make it learn after each step, we only select the best action
            # in the opinion of the agent.
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
    is_terminal = [e == 200 for e in history]
    success = [1. if b else 0. for b in is_terminal]
    print "\n --- Summary ---"
    print "Testing procedure using %d random initialisation steps." % init
    print "Number of failures:", failures
    print "The average testing reward, calculated over %d episodes, is %d" %(episodes - failures,(sum(history)/len(history)))
    print "Success percentage: %f %% \n" % (100*(sum(success)/(episodes - failures)))


env = gym.make('CartPole-v0')
action_size = env.action_space.n
state_size = env.observation_space.high.shape[0]

if useGymTools:
    env = wrappers.Monitor(env, '/Users/baptiste/Programming/RLearning/Cartpole-RMSProp', force = True)

agent = qn.Agent(0.05, 0.00019, 0.99, 256, state_size, action_size, 1048576, 128, 64, False, 0.01)
agent.build_network()
trained_agent, reward_history, q_history = train(agent, env, 1000, 1, 32, False)
print reward_history
tools.rmetric(10, reward_history)
tools.qmetric(q_history)

if useGymTools:
    env.close()
    gym.upload('/Users/baptiste/Programming/RLearning/Cartpole-RMSProp', api_key='sk_oH6439jxQ86N7cPzqpe5tA')

# The initialisation time is chosen to be 10 here : it is the number of steps after
# which the cartpole usually fails when being totally random. This gives the agent
# a particularly difficult situation to make up for.
test(env, trained_agent, 10, 100)
