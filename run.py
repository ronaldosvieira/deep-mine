import marlo
import os
import json
import gym.spaces
import random
import numpy as np
import tensorflow as tf
from skimage import transform
from collections import deque
from PIL import Image
from dqn import DQNetwork

possible_actions = [0, 1, 2, 3, 4]

stack_size = 4

# model hyperparameters
state_size = [84, 84, 4] # four 84x84 frames in the stack
action_size = 5 # move forward/backward, turn left/right
alpha = 0.0002 # learning rate

# training hyperparameters
total_episodes = 1
batch_size = 64

# exp-exp parameters for epsilon-greedy
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001
decay_step = 0

# Q-learning hyperparameters
gamma = 0.95 # discounting rate

# memory hyperparameters
pretrain_length = batch_size # no of experiences stored in memory at first
memory_size = 1000000 # max no of experiences

training = True

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, alpha)

class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

# Instantiate memory
memory = Memory(max_size = memory_size)

# Saver will help us to save our model
saver = tf.train.Saver()

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
        Qs = sess.run(DQNetwork.output, 
            feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]

    return action, explore_probability

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
                                    "client_pool": client_pool
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

    print("Pre-training...")

    # pre train
    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            # First we need a state
            state, stacked_frames = stack_frames(stacked_frames, observation, True)

        # Random action
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        # Look done
        if done:
            # Episode is finished
            next_state = np.zeros(state.shape)

            # Add experience to memory
            memory.add((state, action, reward, next_state, done))

            env.close()

            return
        else:
            next_state = observation
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Add experience to memory
            memory.add((state, action, reward, next_state, done))

            # Our state is now the next_state
            state = next_state

    step = 0
    episode_rewards = []

    state, stacked_frames = stack_frames(stacked_frames, observation, True)

    print("Starting training")

    while True:
        step += 1
        decay_step += 1

        action, explore_probability = predict_action(explore_start, 
            explore_stop, decay_rate, decay_step, state, sess)

        observation, reward, done, info = env.step(action)

        episode_rewards.append(reward)

        if done:
            env.close()

            return
        else:
            next_state, stacked_frames = stack_frames(stacked_frames,
                observation, False)

            memory.add((state, action, reward, next_state, done))

            state = next_state

        ### LEARNING PART
        def ac(a):
            r = [0] * len(possible_actions)
            r[a] = 1
            return r            
        # Obtain random mini-batch from memory
        batch = memory.sample(batch_size)
        states_mb = np.array([each[0] for each in batch], ndmin=3)
        actions_mb = np.array([ac(each[1]) for each in batch])
        rewards_mb = np.array([each[2] for each in batch]) 
        next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        dones_mb = np.array([each[4] for each in batch])

        target_Qs_batch = []

         # Get Q values for next_state 
        Qs_next_state = sess.run(DQNetwork.output, 
            feed_dict = {DQNetwork.inputs_: next_states_mb})

        # Set Q_target = r if the episode ends at s+1, 
        # otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, len(batch)):
            terminal = dones_mb[i]

            # If we are in a terminal state, only equals reward
            if terminal:
                target_Qs_batch.append(rewards_mb[i])
                
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)
                

        targets_mb = np.array([each for each in target_Qs_batch])

        loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                            feed_dict={DQNetwork.inputs_: states_mb,
                                       DQNetwork.target_Q: targets_mb,
                                       DQNetwork.actions_: actions_mb})

        # Write TF Summaries
        summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                           DQNetwork.target_Q: targets_mb,
                                           DQNetwork.actions_: actions_mb})
        writer.add_summary(summary, episode)
        writer.flush()

    print("Sum of rewards:", sum(episode_rewards))

    # It is important to do this env.close()
    env.close()


if __name__ == "__main__":
    """
        In case of debugging locally, run the episode just once
        and in case of when the agent is being evaluated, continue 
        running episodes for as long as the evaluator keeps supplying
        join_tokens.
    """
    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, "./models/model.ckpt")

        if not marlo.is_grading():
            print("Running single episode...")

            # Setup TensorBoard Writer
            writer = tf.summary.FileWriter("tensorboard/DQNetwork/1")

            ## Losses
            tf.summary.scalar("Loss", DQNetwork.loss)

            write_op = tf.summary.merge_all()

            # Initialize the variables
            sess.run(tf.global_variables_initializer())

            for episode in range(total_episodes):
                print("Episode {}".format(episode))

                run_episode(sess, episode)

                if (episode + 1) % 5 == 0:
                    save_path = saver.save(sess, "./models/model.ckpt")
                    print("Model Saved")
        else:
            while True:
                run_episode(sess, episode)
