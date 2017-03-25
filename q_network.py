import gym
import tensorflow as tf
import numpy as np
import random

# We build the Q approximator, using a 2 multilayer perceptron. Weights are
# initialized at random following normal errors, to break symmetries and
# avoid 0 gradient.

class Agent():

    def __init__(self, epsilon, rate, gamma, batch_size, state_size,
        action_size, buffer_size, n_hidden_1, n_hidden_2):
        self.epsilon = epsilon
        self.rate = rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2

        self.buffer = []
        self.steps = 0

    def build_network(self):
        self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.int32, [None])
        self.target_q = tf.placeholder(tf.float32, [None])
        self.q_predict = tf.placeholder(tf.float32, [None])

        self.global_step = tf.Variable(0, trainable = False)

        self.w1 = tf.Variable(tf.random_normal([self.state_size, self.n_hidden_1]))
        self.w2 = tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]))
        self.w3 = tf.Variable(tf.random_normal([self.n_hidden_2, self.action_size]))
        self.b1 = tf.Variable(tf.random_normal([self.n_hidden_1]))
        self.b2 = tf.Variable(tf.random_normal([self.n_hidden_2]))
        self.b3 = tf.Variable(tf.random_normal([self.action_size]))

        layer_1 = tf.add(tf.matmul(self.state_input, self.w1), self.b1)
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, self.w2), self.b2)
        layer_2 = tf.nn.relu(layer_2)

        self.q_values = tf.add(tf.matmul(layer_2, self.w3), self.b3)
        action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
        self.q_predict = tf.reduce_sum(self.q_values * action_mask, 1)

        #learning_rate = tf.train.exponential_decay(self.rate, self.global_step, 10000, 0.95, staircase = True)
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q_predict))
        self.optimizer = tf.train.GradientDescentOptimizer(self.rate) #learning_rate
        self.train_op = self.optimizer.minimize(self.loss, global_step = self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_best_action(self, state):
        actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
        return np.argmax(actions)

    def get_greedy_action(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return self.get_best_action(state)

    def append_memory(self, episode):
        for step in episode:
            self.buffer.append(step)

            if len(self.buffer) > self.buffer_size:
                del self.buffer[:1]

    def decrease_epsilon(self, percent, thresold):
        if self.epsilon > thresold:
            self.epsilon = percent * self.epsilon
        else:
            self.epsilon = thresold

    def learn(self, episode, train_steps):

        self.append_memory(episode)
        for i in range(train_steps):
            self.steps += 1

            # Create the minibatch containing training samples
            if len(self.buffer) > self.batch_size:
                samples = random.sample(self.buffer, self.batch_size)
            else:
                samples = random.sample(self.buffer, len(self.buffer))
            # Next state is stored in s[2]; we want to get the max for all next states in the minibatch.
            q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [s[2] for s in samples]})
            max_q_values = q_values.max(axis = 1)

            # We calculate the target, r + gamma * max Q(s,a) if not done, r else.
            target_q = [(samples[i][3] + self.gamma * max_q_values[i] * (1 - samples[i][4])) for i in range(len(samples))]

            self.sess.run(self.train_op, feed_dict={
                                                # Initial states stored in s[0]
                                                self.state_input: [s[0] for s in samples],
                                                self.target_q: target_q,
                                                # Actions stored in s[1]
                                                self.action: [s[1] for s in samples]
                                                })
