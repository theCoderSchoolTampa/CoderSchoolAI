from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import numpy as np

from CoderSchoolAI.Environment.Attributes import ObsAttribute, ActionAttribute

class Agent:
    def __init__(self, is_user_control=False):
        self.replay_buffer = None
        self.is_user_control = is_user_control
    
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

    def store_memory(self, state, action, probs, vals, reward, done, next_state):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def clear_memory(self):
        raise NotImplementedError("This method should be implemented in a subclass.")

    def size(self,) -> int: # Assuming self.states
        return len(self.probs)


class BasicReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def generate_batches(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                np.array(self.next_states),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done, next_state) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear_memory(self) -> None:
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.next_states = []


class DictReplayBuffer(ReplayBuffer):
    def __init__(self, batch_size, dict_keys=None, act_keys=None):
        self.dict_keys = dict_keys
        self.act_keys = act_keys
        super().__init__(batch_size)

    def generate_batches(self) -> Tuple[Union[Dict, np.ndarray], Union[Dict, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indicies = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self.batch_size] for i in batch_start]
        s = np.array(self.states) if not isinstance(self.states, dict) else {key: np.array(value) for key, value in self.states.items()}
        a =  np.array(self.actions) if not isinstance(self.actions, dict) else {key: np.array(value) for key, value in self.actions.items()}
        ns = np.array(self.next_states) if not isinstance(self.next_states, dict) else {key: np.array(value) for key, value in self.next_states.items()}
        return s, a, \
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                ns,\
                batches

    def store_memory(self, state, action, probs, vals, reward, done, next_state) -> None:
        if self.dict_keys is not None:
            for key, value in state.items():
                self.states[key].append(value)
        else:
            self.states.append(state)
        if self.act_keys is not None:
            for key, value in action.items():
                self.actions[key].append(value)
        else:
            self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

        if self.dict_keys is not None and isinstance(next_state, dict):
            for key, value in next_state.items():
                self.next_states[key].append(value)
        else:
            self.next_states.append(next_state)

    def clear_memory(self) -> None:
        self.states = {key: [] for key in self.dict_keys} if self.dict_keys is not None else []
        self.actions = {key: [] for key in self.act_keys} if self.act_keys is not None else []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.next_states = {key: [] for key in self.dict_keys} if self.dict_keys is not None else []