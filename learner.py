import os
import argparse
import yaml
import threading
import queue
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import _pickle as pickle
import numpy as np

import model
from torch.utils.tensorboard import SummaryWriter

import zmq


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def DQN_train(DQN_policy, DQN_target, DQN_optim, discount_factor, states, actions, rewards, next_states, dones):
    actions = actions.detach().unsqueeze(1).to(device)
    rewards = rewards.detach().unsqueeze(1).to(device)
    dones = dones.detach().unsqueeze(1).to(device)

    x = next_states.detach().to(device)
    with torch.no_grad():
        Q_next = DQN_target(x).max(1)[0].unsqueeze(1).to(device)
    yi = rewards + discount_factor * Q_next * (1 - dones)

    x = states.detach().to(device)
    Q_expected = DQN_policy(x).gather(1, actions)

    loss = nn.functional.mse_loss(yi, Q_expected)
    loss.backward()

    td_error = torch.abs(Q_expected.detach() - yi.detach())
    prios = (td_error + 1e-6).data.cpu().squeeze().numpy()

    return torch.mean(loss), prios


def soft_update_DQN_target(DQN_target, DQN_policy, tau):
    # Soft update of the target network's weights
    # θ′ <- τ θ + (1 − τ)θ′
    target_dict = DQN_target.state_dict()
    policy_dict = DQN_policy.state_dict()
    for key in policy_dict:
        target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
    DQN_target.load_state_dict(target_dict)


def parse_arg():
    parser = argparse.ArgumentParser(description='Ape-X')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--n_actors', type=int)
    parser.add_argument('--replay_ip', type=str)
    args = parser.parse_args()
    return args


def wait_connection(n_actors):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind("tcp://*:52002")
    connected = set()
    finished = set()
    while True:
        identity, null, data = socket.recv_multipart()
        actor_id, signal = pickle.loads(data)
        socket.send_multipart((identity, null, b''))
        if signal == 1:
            connected.add(actor_id)
            print(f"Received handshake signal from actor {actor_id}")
        else:
            finished.add(actor_id)
            print(f"connection established: {actor_id}")

        if len(connected) == (n_actors + 1) and (len(connected) == len(finished)):
            break

    socket.close()
    ctx.term()
    print("Successfully connected with all actors!")


def send_param(param_queue):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.set_hwm(3)
    socket.bind("tcp://*:52001")
    while True:
        print("waiting param to send...")
        param = param_queue.get()
        print("sending param...")
        data = pickle.dumps(param)
        socket.send(data, copy=False)
        param, data = None, None


def recv_batch(batch_queue, replay_ip):
    def _thunk(thread_queue):
        ctx = zmq.Context.instance()
        socket = ctx.socket(zmq.DEALER)
        socket.setsockopt(zmq.IDENTITY, pickle.dumps(f'dealer-{os.getpid()}'))
        socket.connect(f"tcp://{replay_ip}:51003")
        while True:
            socket.send(b'')
            data = socket.recv(copy=False)
            thread_queue.put(data)

    thread_queue = queue.Queue(maxsize=3)
    threading.Thread(target=_thunk, args=(thread_queue, )).start()

    while True:
        print("[recv_batch] waiting batch queue...")
        data = thread_queue.get()
        batch = pickle.loads(data)

        keys, states, actions, rewards, next_states, dones = batch
        states = np.array([np.array(state) for state in states])
        next_states = np.array([np.array(state) for state in next_states])

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        batch = [keys, states, actions, rewards, next_states, dones]

        batch_queue.put(batch)
        data, batch = None, None


def send_prios(prios_queue, replay_ip):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.DEALER)
    socket.connect(f"tcp://{replay_ip}:51002")

    while True:
        keys, prios = prios_queue.get()
        socket.send(pickle.dumps((keys, prios)), copy=False)
        socket.recv()
        keys, prios = None, None


def train(args, config, batch_queue, prios_queue, param_queue):
    writer = SummaryWriter(log_dir=config["tb_log_path"])
    tau = config["learner_tau"]
    # Init meta variables
    DQN_policy = model.DuelingDQN().to(device)
    DQN_target = model.DuelingDQN().to(device)

    DQN_target.load_state_dict(DQN_policy.state_dict())
    DQN_optim = optim.Adam(DQN_policy.parameters(), lr=config["learning_rate"])

    learning_step = 0

    wait_connection(args.n_actors)

    state_dict = DQN_target.state_dict()
    cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
    param_queue.put(cpu_state_dict)

    print("start learning ...")
    while True:
        print("step: ", learning_step)
        keys, states, actions, rewards, next_states, dones = batch_queue.get()

        DQN_optim.zero_grad()
        loss_mean, prios = DQN_train(DQN_policy, DQN_target, DQN_optim, config["discount_factor"], states, actions, rewards, next_states, dones)
        DQN_optim.step()

        prios_queue.put((keys, prios))

        soft_update_DQN_target(DQN_target, DQN_policy, tau)

        learning_step += 1

        if learning_step % 5 == 0:
            print("[train] write loss")
            writer.add_scalar('meta/LossDQN', loss_mean, learning_step)

        if learning_step % config["save_param_interval"] == 0:
            print("[train] save state_dict")
            torch.save(DQN_target.state_dict(), config["out_path"] + f"model-{learning_step}.pth")

        if learning_step % config["publish_param_interval"] == 0:
            print("[train] send state_dict")
            state_dict = DQN_target.state_dict()
            cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
            param_queue.put(cpu_state_dict)

        keys, states, actions, rewards, next_states, dones = None, None, None, None, None, None


def main():
    mp.set_start_method('spawn')

    args = parse_arg()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    batch_queue = mp.Queue(maxsize=config['batch_queue_size'])
    prios_queue = mp.Queue(maxsize=config['prios_queue_size'])
    param_queue = mp.Queue(maxsize=config['param_queue_size'])

    procs = [
        mp.Process(target=train, args=(args, config, batch_queue, prios_queue, param_queue)),
        mp.Process(target=send_param, args=(param_queue, )),
        mp.Process(target=send_prios, args=(prios_queue, args.replay_ip)),
    ]
    procs += [
        mp.Process(target=recv_batch, args=(batch_queue, args.replay_ip))
        for _ in range(config['n_recv_batch_process'])
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
