import marlo
import os
import json
import gym.spaces
import numpy as np
from skimage import transform
from collections import deque
from PIL import Image

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

def run_episode():
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

    # Enter game loop
    done = False
    while not done:
        preprocessed_frame = preprocess_frame(observation)
        
        _action = env.action_space.sample()

        observation, reward, done, info = env.step(_action)
        
        print("reward:", reward)
        print("done:", done)
        print("info", info)

    # It is important to do this env.close()
    env.close()


if __name__ == "__main__":
    """
        In case of debugging locally, run the episode just once
        and in case of when the agent is being evaluated, continue 
        running episodes for as long as the evaluator keeps supplying
        join_tokens.
    """
    if not marlo.is_grading():
        print("Running single episode...")
        run_episode()
    else:
        while True:
            run_episode()
