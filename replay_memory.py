import numpy as np
import argparse
import yaml
import asyncio
import torch
import concurrent
import pickle


from collections import namedtuple, OrderedDict

import zmq
from zmq.asyncio import Context

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'q', 'next_q'])
recv_batch_uri_outer = "tcp://*:51001"
recv_batch_uri_inner = "ipc:///tmp/5101.ipc"
recv_prios_uri_outer = "tcp://*:51002"
recv_prios_uri_inner = "ipc:///tmp/5102.ipc"
send_batch_uri_outer = "tcp://*:51003"
send_batch_uri_inner = "ipc:///tmp/5103.ipc"


class ReplayMemory:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.memory = OrderedDict()
        self.priorities = OrderedDict()
        self.probabilities = OrderedDict()
        self.probabilities_updated = False

        self.alpha = alpha

    def _calc_probabilities(self, priorities):
        total_priority_sum = sum(priorities.values())
        probabilities = [p ** self.alpha / total_priority_sum for p in priorities.values()]

        total_probability_sum = sum(probabilities)
        return OrderedDict([(k, v / total_probability_sum) for k, v in zip(priorities.keys(), probabilities)])

    def update_probabilities(self):
        self.probabilities_updated = True
        self.probabilities = self._calc_probabilities(self.priorities)

    def add(self, key, state, action, reward, next_state, done, q, next_q, prio):
        if len(self.memory) == self.capacity:
            self.memory.popitem(last=False)
            self.priorities.popitem(last=False)
            self.probabilities.popitem(last=False)

        self.memory[key] = Experience(state, action, reward, next_state, done, q, next_q)
        self.priorities[key] = prio
        self.probabilities_updated = False

    def sample(self, batch_size):
        assert self.probabilities_updated, "probabilities not updated"

        sampled_keys = np.random.choice(list(self.priorities.keys()), size=batch_size, p=list(self.probabilities.values()))
        encoded_sample = self._encode_sample(sampled_keys)

        return tuple([sampled_keys] + list(encoded_sample))

    def _encode_sample(self, keys):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for key in keys:
            assert key in self.memory, 'invalid key provided'
            data = self.memory[key]
            state, action, reward, next_state, done, _, _ = data
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def update_priorities(self, keys, priorities):
        assert len(keys) == len(priorities), f'{len(keys)}, {len(priorities)}'
        for key, priority in zip(keys, priorities):
            assert priority > 0, f"invalid priority: {priority}"
            if key in self.priorities:
                self.priorities[key] = priority

        self.update_probabilities()

    @property
    def size(self):
        return len(self.memory)


def recv_batch_proxy():
    ctx = zmq.Context()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind(recv_batch_uri_outer)
    backend = ctx.socket(zmq.DEALER)
    backend.bind(recv_batch_uri_inner)
    print("waiting for connection to:", recv_batch_uri_outer)
    zmq.proxy(frontend, backend)


def recv_prios_proxy():
    ctx = zmq.Context()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind(recv_prios_uri_outer)
    backend = ctx.socket(zmq.DEALER)
    backend.bind(recv_prios_uri_inner)
    print("waiting for connection to:", recv_prios_uri_outer)
    zmq.proxy(frontend, backend)


def send_batch_proxy():
    ctx = zmq.Context()
    frontend = ctx.socket(zmq.ROUTER)
    frontend.bind(send_batch_uri_outer)
    backend = ctx.socket(zmq.DEALER)
    backend.bind(send_batch_uri_inner)
    print("waiting for connection to:", send_batch_uri_outer)
    zmq.proxy(frontend, backend)


def push_batch(buffer, data):
    batch, prios = pickle.loads(data)
    for sample in zip(*batch, prios):
        buffer.add(*sample)
    buffer.update_probabilities()
    print("len:", buffer.size)


def update_prios(buffer, data):
    keys, prios = pickle.loads(data)
    buffer.update_priorities(keys, prios)


def sample_batch(buffer, batch_size):
    batch = buffer.sample(batch_size)
    data = pickle.dumps(batch)
    return data


async def recv_batch_worker(buffer, exe, lock):
    loop = asyncio.get_event_loop()
    ctx = Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect(recv_batch_uri_inner)

    while True:
        identity, data = await socket.recv_multipart(copy=False)
        async with lock:
            await loop.run_in_executor(exe, push_batch, buffer, data)
        await socket.send_multipart((identity, b''))


async def recv_prios_worker(buffer, exe, lock):
    loop = asyncio.get_event_loop()
    ctx = Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect(recv_prios_uri_inner)
    while True:
        identity, data = await socket.recv_multipart(copy=False)
        async with lock:
            await loop.run_in_executor(exe, update_prios, buffer, data)
        await socket.send_multipart((identity, b''))


async def send_batch_worker(buffer, exe, lock, batch_size):
    loop = asyncio.get_event_loop()
    ctx = Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect(send_batch_uri_inner)
    while True:
        identity, _ = await socket.recv_multipart(copy=False)

        while True:
            async with lock:
                if buffer.size > batch_size:
                    break

            await asyncio.sleep(1)

        async with lock:
            batch = await loop.run_in_executor(exe, sample_batch, buffer, batch_size)
        await socket.send_multipart([identity, batch], copy=False)


def parse_arg():
    parser = argparse.ArgumentParser(description='Ape-X')
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    return args


async def main():
    args = parse_arg()
    if args.config_file is not None:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print("Error: Config file path is not specified.")
        exit()

    procs = [torch.multiprocessing.Process(target=func) for func in [recv_batch_proxy, recv_prios_proxy, send_batch_proxy]]
    [p.start() for p in procs]

    buffer = ReplayMemory(config['replay_buffer_size'], config['replay_buffer_alpha'])

    exe = concurrent.futures.ThreadPoolExecutor()
    lock = asyncio.Lock()

    workers = [recv_batch_worker(buffer, exe, lock),
               recv_prios_worker(buffer, exe, lock),
               send_batch_worker(buffer, exe, lock, config['batch_size'])]

    await asyncio.gather(*workers)
    return True


if __name__ == '__main__':
    asyncio.run(main())
