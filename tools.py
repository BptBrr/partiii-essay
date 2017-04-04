import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

# rmetric plots the average total rewards, taking as input a stride and the
# rewards from the learning session. The stride corresponds to the number of
# rewards averaged. We use a sliding window of length 'stride', average the
# rewards in the window and print the resulting list.
def rmetric(stride, rewards):
    n = len(rewards)
    k = n/stride
    avg = []
    for i in range(n):
        avg.append(np.mean(rewards[i:i+stride]))
    plt.plot(avg)
    plt.ylabel('Average total reward')
    plt.xlabel('Episodes')
    plt.show()

# qmetric plots its input, the q_history. Maximum average q values are computed
# after each episode, during the training loop.
def qmetric(q_history):
    plt.plot(q_history)
    plt.ylabel('Maximum average Q value')
    plt.xlabel('Episodes')
    plt.show()
