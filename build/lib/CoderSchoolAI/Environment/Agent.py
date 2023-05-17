from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import numpy as np

from CoderSchoolAI.Environment.Attributes import ObsAttribute, ActionAttribute

class Agent:
    def __init__(self):
        self.replay_buffer = None
    
    def get_actions(self):
        """
        Returns the list of actions that the agent can take in the current state of the environment.
        """
        raise NotImplementedError("This method should be implemented in a subclass, not the Base Class Agent.")

    def get_next_action(self, state):
        """
        This method should be implemented in a subclass to return the next action to be taken by the agent
        based on the current state of the environment.
        """
        raise NotImplementedError("This method should be implemented in a subclass, not the Base Class Agent.")

    def update(self, state, action, next_state, reward):
        """
        This method should be implemented in a subclass to update the agent's knowledge or parameters based on the
        observed state, action, next state, and reward from the environment.
        """
        raise NotImplementedError("This method should be implemented in a subclass, not the Base Class Agent.")

class ReplayBuffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.clear_memory()

    def generate_batches(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def store_memory(self, state, action, probs, vals, reward, done):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def clear_memory(self):
        raise NotImplementedError("This method should be implemented in a subclass.")


class BasicReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indicies = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class DictReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indicies = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self.batch_size] for i in batch_start]

        return {key: np.array(value) for key, value in self.states.items()},\
                {key: np.array(value) for key, value in self.actions.items()},\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        for key, value in state.items():
            self.states[key].append(value)
        for key, value in action.items():
            self.actions[key].append(value)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = {key: [] for key in self.states.keys()}
        self.actions = {key: [] for key in self.actions.keys()}
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []