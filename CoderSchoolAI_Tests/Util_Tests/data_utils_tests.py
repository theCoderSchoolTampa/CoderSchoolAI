from CoderSchoolAI.Util.data_utils import *
import torch as th
from typing import Union, Dict
import pytest  # for exception testing

def test_get_minibatches():
    # Test valid tensor inputs
    for dtype in ['cpu', 'cuda:0'] if th.cuda.is_available() else ['cpu']:
        device = th.device(dtype)

        states = th.randn(100, 4, device=device)
        probs = th.rand(100, 5, device=device)
        returns = th.randn(100, 1, device=device)
        advantages = th.randn(100, 1, device=device)

        n_minibatches = 5
        mb_size = 20

        minibatches = get_minibatches(states, probs, returns, advantages, n_minibatches, mb_size, device)

        assert len(minibatches) == n_minibatches, f"Expected {n_minibatches} minibatches, got {len(minibatches)}"

        for mb_states, mb_probs, mb_returns, mb_advantages in minibatches:
            assert mb_states.shape[0] == mb_size, f"Expected minibatch size {mb_size}, got {mb_states.shape[0]}"
            assert mb_states.device == device, f"Expected device {device}, got {mb_states.device}"

    # Test valid dictionary inputs
    states = {'state1': th.randn(100, 4), 'state2': th.randn(100, 3)}

    minibatches = get_minibatches(states, probs, returns, advantages, n_minibatches, mb_size, device)
    for mb_states, _, _, _ in minibatches:
        for key in states.keys():
            assert key in mb_states, f"Key {key} missing in minibatch states"
            assert mb_states[key].shape[0] == mb_size, f"Expected minibatch size {mb_size}, got {mb_states[key].shape[0]}"

    # # Test invalid inputs (e.g., mismatched shapes)
    # with pytest.raises(Exception):
    #     get_minibatches(th.randn(99, 4), probs, returns, advantages, n_minibatches, mb_size, device)

    # with pytest.raises(Exception):
    #     get_minibatches({'state1': th.randn(99, 4)}, probs, returns, advantages, n_minibatches, mb_size, device) TODO: Implement assertions in data_utils

    print("All tests passed.")

# Run the test

if __name__ == '__main__':
    test_get_minibatches()
