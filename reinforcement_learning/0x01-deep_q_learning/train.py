#!/usr/bin/env python3
"""train an agent that can play Atari’s Breakout"""
from PIL import Image
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
from rl.agents.dnq import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


class AtariProcessor(Processor):
    """Atari preprocessor"""
    def process_observation(self, observation):
        """process observation"""
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == (84, 84)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """process state batch"""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """process reward"""
        return np.clip(reward, -1., 1.)


def build_model(num_action):
    """build model"""
    input_shape = (4,) + (84, 84)
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_action))
    model.add(Activation('linear'))
    return model(outputs=num_action)


if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    env.reset()
    num_action = env.action_space.n
    window = 4
    model = build_model(num_action)
    model.summary()
    memory = SequentialMemory(limit=1000000, window_length=4)
    processor = AtariProcessor()
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1., value_min=.1, value_test=.05,
                                  nb_steps=1000000)

    dqn = DQNAgent(model=model, nb_actions=num_action, policy=policy,
                   memory=memory, processor=processor,
                   nb_steps_warmup=50000, gamma=.99,
                   target_model_update=10000,
                   train_interval=4,
                   delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    dqn.fit(env,
            nb_steps=17500,
            log_interval=10000,
            visualize=False,
            verbose=2)

    dqn.save_weights('policy.h5', overwrite=True)
