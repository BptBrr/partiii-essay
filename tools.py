import gym
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

def rmetric(stride, rewards):
    n = len(rewards)
    k = n/stride
    avg = []
    for i in range(k):
        avg.append(np.mean(rewards[i*stride:(i+1)*stride]))
    avg.append(np.mean(rewards[k*stride+1:]))
    plt.plot(avg)
    plt.ylabel('Average total reward')
    plt.show()

#def qmetric():
