#!/usr/bin/env python3
"""train an agent that can play Atari’s Breakout"""
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
env = gym.make("Breakout-v0")
nb_actions = env.action_space.n
input_shape = env.observation_space.shape


def build_model(input_shape, nb_actions):
    """Build a convolutional network model"""
    model = Sequential()
    model.add(Input(shape=(3,) + input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


model = build_model(input_shape, nb_actions)


def build_agent(model, nb_actions):
    """Builds agent"""
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1,
        value_min=.1,
        value_test=.2,
        nb_steps=10000000
    )
    memory = SequentialMemory(
        limit=10000000,
        window_length=3
    )
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        enable_dueling_network=True,
        dueling_type='avg',
        nb_actions=nb_actions,
        nb_steps_warmup=50000
    )
    dqn.compile(Adam(learning_rate=1e-4), metrics=['mae'])
    return dqn


dqn = build_agent(model, nb_actions)


def train_agent(dqn, env, weights_filename):
    """Trains agent"""
    checkpoint_weights_filename = 'dqn_Breakout-v0' + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format("Breakout-v0")
    callbacks = [
        ModelIntervalCheckpoint(
            checkpoint_weights_filename,
            interval=250000
        )
    ]
    callbacks += [FileLogger(log_filename, interval=1000)]
    dqn.fit(
        env,
        callbacks=callbacks,
        nb_steps=10000000,
        log_interval=1000
    )
    dqn.save_weights(
        weights_filename,
        overwrite=True
    )
