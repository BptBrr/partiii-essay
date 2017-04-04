import gym
import tensorflow as tf
import numpy as np
import random

# We build the Q approximator, using a 2 multilayer perceptron. Weights are
# initialized at random following normal errors, to break symmetries and
# avoid 0 gradient.

# We define here the Agent class, with functions allowing him to learn using the
# Deep Q-Learning algorithm.

class Agent():

    def __init__(self, epsilon, rate, gamma, batch_size, state_size,
        action_size, buffer_size, n_hidden_1, n_hidden_2, anneal, end_eps):
        # Probability of random action in epsilon-greedy strategy :
        self.epsilon = epsilon
        # Learning rate :
        self.rate = rate
        # Discount rate :
        self.gamma = gamma
        # Minibatch size :
        self.batch_size = batch_size
        # Sizes of action and state spaces :
        self.state_size = state_size
        self.action_size = action_size
        # Size of the replay memory :
        self.buffer_size = buffer_size
        # Number of neurons in hidden layers :
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        # Boolean controlling the annealing of epsilon :
        self.anneal = anneal
        # When annealing, epsilon thresold :
        self.end_eps = end_eps

        self.buffer = []
        self.steps = 0

    # build_network builds the multilayer perceptron, using TensorFlow architecture.
    def build_network(self):
        self.state_input = tf.placeholder(tf.float32, [None, self.state_size])
        self.action = tf.placeholder(tf.int32, [None])
        self.target_q = tf.placeholder(tf.float32, [None])
        self.q_predict = tf.placeholder(tf.float32, [None])

        # global_step allows to use a decaying of the learning rate. The associated
        # learning rate is defined below, as learning_rate.
        self.global_step = tf.Variable(0, trainable = False)

        # Weights and biases of the network.
        self.w1 = tf.Variable(tf.random_normal([self.state_size, self.n_hidden_1]))
        self.w2 = tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2]))
        self.w3 = tf.Variable(tf.random_normal([self.n_hidden_2, self.action_size]))
        self.b1 = tf.Variable(tf.random_normal([self.n_hidden_1]))
        self.b2 = tf.Variable(tf.random_normal([self.n_hidden_2]))
        self.b3 = tf.Variable(tf.random_normal([self.action_size]))

        # Layers definition. We put a ReLU activation function between layers.
        layer_1 = tf.add(tf.matmul(self.state_input, self.w1), self.b1)
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, self.w2), self.b2)
        layer_2 = tf.nn.relu(layer_2)

        self.q_values = tf.add(tf.matmul(layer_2, self.w3), self.b3)

        # q_predict corresponds to the Q-values corresponding to the actions we
        # took. We use a one_hot mask to get, for each action, a list containing
        # a 1 where the action taken is, and 0 everywhere else. q_predict outputs
        # the values for these actions.
        action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
        self.q_predict = tf.reduce_sum(self.q_values * action_mask, 1)

        #learning_rate = tf.train.exponential_decay(self.rate, self.global_step, 10000, 0.95, staircase = True)

        # We define here the loss and optimiser parts of the optimisation problem.
        # The loss is the MSE between target and prediction, and the optimiser is RMSProp.
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q_predict))
        self.optimizer = tf.train.RMSPropOptimizer(self.rate, decay = 0.99)
        self.train_op = self.optimizer.minimize(self.loss, global_step = self.global_step)

        # Launch the session and the initialisation of the network.
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # get_best_action : from a state, returns the action with maximal Q-value.
    def get_best_action(self, state):
        actions = self.sess.run(self.q_values, feed_dict={self.state_input: [state]})
        return np.argmax(actions)

    # get_greedy_action implements the epsilon greedy strategy, using get_best_action.
    def get_greedy_action(self,state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return self.get_best_action(state)

    # get_max_q is used to build the Q-metric. It outputs the maximum Q-values
    # from a given set of states.
    def get_max_q(self,states):
        predicted_q = self.sess.run(self.q_values, feed_dict={self.state_input: states})
        return predicted_q.max(axis = 1)

    # append_memory adds a step to the replay memory. If the buffer is full,
    # it discards the first recorded step.
    def append_memory(self, step):
        self.buffer.append(step)

        if len(self.buffer) > self.buffer_size:
                del self.buffer[:1]

    # decrease_epsilon implements the annealing of espilon. Percent is the decreasing
    # percentage at each step.
    def decrease_epsilon(self, percent):
        if self.epsilon > self.end_eps:
            self.epsilon = (1 - percent) * self.epsilon
        else:
            self.epsilon = self.end_eps

    def learn(self, epochs):

        for i in range(epochs):
            self.steps += 1

            # Create the minibatches containing training samples
            if len(self.buffer) > self.batch_size:
                samples = random.sample(self.buffer,self.batch_size)
            else:
                samples = random.sample(self.buffer, len(self.buffer))

            # Next state is stored in s[2]; we want to get the max for all next states in the minibatch.
            q_values = self.sess.run(self.q_values, feed_dict={self.state_input: [s[2] for s in samples]})
            max_q_values = q_values.max(axis = 1)

            # We calculate the target, r + gamma * max Q(s,a) if not done (i.e. not terminal), r else.
            target_q = [(samples[i][3] + self.gamma * max_q_values[i] * (1 - samples[i][4])) for i in range(len(samples))]

            self.sess.run(self.train_op, feed_dict={
                                                # Initial states stored in s[0]
                                                self.state_input: [s[0] for s in samples],
                                                self.target_q: target_q,
                                                # Actions stored in s[1]
                                                self.action: [s[1] for s in samples]
                                                })
