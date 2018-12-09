# -*- coding: utf-8 -*-
import marlo
import os
import json
import gym.spaces
import random
import numpy as np
import tensorflow as tf
from skimage import transform
from collections import deque
from operator import attrgetter
from PIL import Image
from dqn import DQNetwork
from ga import *

training = True

possible_actions = [1, 2, 3, 4]

stack_size = 4

# model hyperparameters
state_size = [84, 84, 4] # four 84x84 frames in the stack
action_size = 5 # move forward/backward, turn left/right
alpha = 0.0002 # learning rate

# training hyperparameters
total_episodes = 50
batch_size = 64

# exp-exp parameters for epsilon-greedy
explore_start = 1.0 if training else 0.0
explore_stop = 0.01 if training else 0.0
decay_rate = 0.0001
decay_step = 0

# Q-learning hyperparameters
gamma = 0.95 # discounting rate

# memory hyperparameters
pretrain_length = batch_size # no of experiences stored in memory at first
memory_size = 1000000 # max no of experiences

sigma = 0.1
I = 1
T = 100

max_seed = 1000000

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
dqn = DQNetwork(state_size, action_size, alpha)

total_parameters = 0
variables = tf.trainable_variables()

for var in variables:
    shape = var.get_shape()

    var_parameters = 1

    for dim in shape:
        var_parameters *= dim.value

    total_parameters += var_parameters


def gen_initial_weights(seed):
    np.random.seed(seed)

    w = np.array([
        np.random.normal(0, np.sqrt(2 / 2320), size = variables[0].get_shape()),
        np.zeros(shape = variables[1].get_shape()),
        np.ones(shape = variables[2].get_shape()),
        np.zeros(shape = variables[3].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1542), size = variables[4].get_shape()),
        np.zeros(shape = variables[5].get_shape()),
        np.ones(shape = variables[6].get_shape()),
        np.zeros(shape = variables[7].get_shape()),
        np.random.normal(0, np.sqrt(2 / 3078), size = variables[8].get_shape()),
        np.zeros(shape = variables[9].get_shape()),
        np.ones(shape = variables[10].get_shape()),
        np.zeros(shape = variables[11].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1664), size = variables[12].get_shape()),
        np.zeros(shape = variables[13].get_shape()),
        np.random.normal(0, np.sqrt(2 / 517), size = variables[14].get_shape()),
        np.zeros(shape = variables[15].get_shape())
    ])

    np.random.seed(None)

    return w

def gen_weights(seed):
    np.random.seed(seed)

    w = np.array([
        np.random.normal(0, np.sqrt(2 / 2320), 
            size = variables[0].get_shape()),
        np.random.normal(0, np.sqrt(2 / 2320), 
            size = variables[1].get_shape()),
        np.random.normal(0, np.sqrt(2 / 2320), 
            size = variables[2].get_shape()),
        np.random.normal(0, np.sqrt(2 / 2320), 
            size = variables[3].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1542), 
            size = variables[4].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1542), 
            size = variables[5].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1542), 
            size = variables[6].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1542), 
            size = variables[7].get_shape()),
        np.random.normal(0, np.sqrt(2 / 3078), 
            size = variables[8].get_shape()),
        np.random.normal(0, np.sqrt(2 / 3078), 
            size = variables[9].get_shape()),
        np.random.normal(0, np.sqrt(2 / 3078), 
            size = variables[10].get_shape()),
        np.random.normal(0, np.sqrt(2 / 3078), 
            size = variables[11].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1664), 
            size = variables[12].get_shape()),
        np.random.normal(0, np.sqrt(2 / 1664), 
            size = variables[13].get_shape()),
        np.random.normal(0, np.sqrt(2 / 517), 
            size = variables[14].get_shape()),
        np.random.normal(0, np.sqrt(2 / 517), 
            size = variables[15].get_shape())
    ])

    np.random.seed(None)

    return w

class ParameterVector(Individual):
    def decode(self):
        var_weights = gen_initial_weights(self.genotype[0])

        for seed in self.genotype[1:]:
            var_weights += sigma * gen_weights(seed)

        return var_weights

    def evaluate(self, amount = 1):
        var_weights = self.decode()
        rewards = []

        for weight, variable in zip(var_weights, variables):
            sess.run(variable.assign(weight))

        for episode in range(amount):
            rewards.append(run_episode(sess, episode))

        self.fitness = np.mean(rewards)
        print(self.fitness)

    def __str__(self):
        return str(self.genotype)

    def __repr__(self):
        return str(self)

def new_individual():
    return ParameterVector([np.random.randint(max_seed)])

def select(pop, amount = 1):
    top_T = sorted(pop, reverse = False, key = attrgetter('fitness'))[:T]
    
    return np.random.choice(top_T, amount)

def mutation(ind):
    return ParameterVector(ind.genotype + [np.random.randint(max_seed)])

def crossover(ind1, ind2):
    return ind1

def preprocess_frame(frame):
    # convert to grayscale
    frame = np.mean(frame, -1)
    
    # normalize pixel values
    normalized_frame = frame / 255.0
    
    # resize down to 84x84
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    # saves resized image for debugging purposes
    # image = Image.new('L', (84, 84))
    # image.putdata(preprocessed_frame.flatten() * 255)
    # image.save("img.png")
    
    return preprocessed_frame

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype = np.int) 
            for i in range(stack_size)], maxlen = 4)

        # Episode is new, so copy the same frame 4x
        for i in range(4):
            stacked_frames.append(frame)
    else:
        # Append frame to deque
        stacked_frames.append(frame)

    # Build the stacked state
    stacked_state = np.stack(stacked_frames, axis = 2)

    return stacked_state, stacked_frames

def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, sess):
    # e-greedy strategy
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) \
        * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Random action (explore)
        action = random.choice(possible_actions)
    else:
        # Get action from DQN (exploit)

        # Estimate Q values
        Qs = sess.run(dqn.output, 
            feed_dict = {dqn.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability

def int_to_one_hot(action):
    encoding = [0] * len(possible_actions)
    encoding[action] = 1

    return encoding

def get_join_tokens():
    if marlo.is_grading():
        """
            In the crowdAI Evaluation environment obtain the join_tokens 
            from the evaluator
            
            the `params` parameter passed to the `evaluator_join_token` only allows
            the following keys : 
                    "seed",
                    "tick_length",
                    "max_retries",
                    "retry_sleep",
                    "step_sleep",
                    "skip_steps",
                    "videoResolution",
                    "continuous_to_discrete",
                    "allowContinuousMovement",
                    "allowDiscreteMovement",
                    "allowAbsoluteMovement",
                    "add_noop_command",
                    "comp_all_commands"
                    # TODO: Add this to the official documentation ? 
                    # Help Wanted :D Pull Requests welcome :D 
        """
        join_tokens = marlo.evaluator_join_token(params={})

    else:
        """
            When debugging locally,
            Please ensure that you have a Minecraft client running on port 10000
            by doing : 
            $MALMO_MINECRAFT_ROOT/launchClient.sh -port 10000
        """
        client_pool = [('127.0.0.1', 10000)]
        join_tokens = marlo.make('MarLo-FindTheGoal-v0',
                                 params={
                                    "client_pool": client_pool,
                                    'tick_length': 25,
                                    'prioritise_offscreen_rendering': False
                                 })
    return join_tokens

def run_episode(sess, episode):
    global decay_step
    """
    Single episode run
    """
    join_tokens = get_join_tokens()

    # As this is a single agent scenario,there will just be a single token
    assert len(join_tokens) == 1
    join_token = join_tokens[0]
    #
    # # Initialize the environment
    env = marlo.init(join_token)

    # Get the first observation
    observation = env.reset()

    stacked_frames = deque([np.zeros((84, 84), dtype = np.int) 
        for i in range(stack_size)], maxlen = 4)

    # Enter game loop
    done = False

    '''if training:
        # pre train
        for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                # First we need a state
                state, stacked_frames = stack_frames(stacked_frames, observation, True)

            # Null action
            observation, reward, done, info = env.step(0)

            # Look done
            if done:
                env.close()

                return run_episode(sess, episode)
            else:
                next_state = observation
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Our state is now the next_state
                state = next_state'''

    step = 0
    episode_rewards = []

    state, stacked_frames = stack_frames(stacked_frames, observation, True)

    while not done:
        step += 1

        # chooses action
        Qs = sess.run(dqn.output, 
            feed_dict = {dqn.inputs_: state.reshape((1, *state.shape))})
        action = np.argmax(Qs)
        action = possible_actions[int(action)]

        # does action
        observation, reward, done, info = env.step(action)

        # saves reward value
        episode_rewards.append(reward)

        # stacks new frame
        next_state, stacked_frames = stack_frames(stacked_frames,
            observation, False)

        # st+1 is now our current state
        state = next_state

    # Get the total reward of the episode
    total_reward = np.sum(episode_rewards)

    '''print('Episode: {}'.format(episode),
          'Total reward: {}'.format(total_reward),
          'Training loss: {:.4f}'.format(loss),
          'Explore P: {:.4f}'.format(explore_probability))'''

    # It is important to do this env.close()
    env.close()

    return total_reward

if __name__ == "__main__":
    """
        In case of debugging locally, run the episode just once
        and in case of when the agent is being evaluated, continue 
        running episodes for as long as the evaluator keeps supplying
        join_tokens.
    """
    with tf.Session() as sess:
        if not marlo.is_grading():
            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            trainer = GeneticAlgorithm(new_individual, select, crossover, mutation)

            best, info = trainer.run(N = 2, G = 2, elitism = 1)

            print(info)
            print(best)
        else:
            while True:
                run_episode(sess, episode)
