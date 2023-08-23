import random
import numpy as np
from collections import namedtuple
from replay_memory import ReplayMemory
from collections import Counter

# Define the Experience class for testing
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'q', 'next_q'])


def test_replay_memory():
    # Set the parameters
    capacity = 100
    alpha = 0.6
    batch_size = 32

    # Create an instance of ReplayMemory
    memory = ReplayMemory(capacity, alpha)

    # Create dummy data
    keys = ['key1', 'key2', 'key3', 'key4']
    states = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12])]
    actions = [0, 1, 2, 0]
    rewards = [0.1, 0.2, 0.3, 0.4]
    next_states = [np.array([2, 3, 4]), np.array([5, 6, 7]), np.array([8, 9, 10]), np.array([11, 12, 13])]
    dones = [False, False, True, False]
    q_values = [0.5, 0.6, 0.7, 0.8]
    next_q_values = [0.9, 1.0, 1.1, 1.2]
    priorities = [0.01, 0.1, 1, 10]

    # Test data addition and sampling
    for i in range(len(keys)):
        memory.add(keys[i], states[i], actions[i], rewards[i], next_states[i], dones[i], q_values[i], next_q_values[i], priorities[i])
    memory.update_probabilities()

    assert memory.size == len(keys), "Size of memory should be equal to the number of added keys."

    # Test sampling
    sampled_data = memory.sample(batch_size)
    assert len(sampled_data) == 6, "Sampled data should contain 6 elements."
    assert len(sampled_data[0]) == batch_size, "Sampled keys should have length equal to the batch size."

    # count key
    counter = Counter(sampled_data[0])

    # calculate probabilities
    total = len(keys)
    probabilities = {key: count / total for key, count in counter.items()}

    # print probs
    for key, probability in probabilities.items():
        print(f'{key}: {probability:.2f}')

    # Test updating priorities and resampling
    updated_priorities = [0.6, 0.7, 0.8, 0.9]
    memory.update_priorities(keys, updated_priorities)

    assert all(memory.priorities[key] == prio for key, prio in zip(keys, updated_priorities)), "Priorities should be updated."
    assert memory.probabilities_updated, "Probabilities should be updated after updating priorities."
    # Test resampling with the added data and updated priorities
    resampled_data = memory.sample(batch_size)
    assert len(resampled_data) == 6, "Resampled data should contain 6 elements."
    assert len(resampled_data[0]) == batch_size, "Resampled keys should have length equal to the batch size."


def add_large_data(replay_memory, num_data):
    for i in range(num_data):
        state = random.randint(0, 9)
        action = random.randint(0, 2)
        reward = random.uniform(0.0, 1.0)
        next_state = random.randint(0, 9)
        done = random.choice([True, False])
        q = random.uniform(0.0, 1.0)
        next_q = random.uniform(0.0, 1.0)
        prio = random.uniform(0.0, 1.0)

        key = f'data_{i}'
        replay_memory.add(key, state, action, reward, next_state, done, q, next_q, prio)
        replay_memory.update_probabilities()
        print(replay_memory.size)


def test_large_data():
    capacity = 10000
    alpha = 0.5
    replay_memory = ReplayMemory(capacity, alpha)
    num_data = 1000000
    add_large_data(replay_memory, num_data)


if __name__ == "__main__":
    test_replay_memory()
    test_large_data()
