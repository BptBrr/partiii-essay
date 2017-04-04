# partiii-essay
Code for my Cambridge Part III Essay, about Deep Reinforcement Learning. The code is written in Python, and uses TensorFlow and OpenAI Gym.
This repo contains four files, all commented (except for mountaincar, which follows exactly the same ideas than cartpole).

**q_network.py** contains the implementation of the optimisation problem (see essay for more info). The Agent class and all its methods implements the deep q-learning algorithm. **cartpole.py** contains the solving of the cartpole environment, and a custom test function. **mountaincar.py** contains the solving of the mountaincar environment. **tools.py** provides some plotting tools to assess the training of agents. The two metrics used are the average total rewards and the average maximum predicted Q values.

**Last update** : 4/04/17
