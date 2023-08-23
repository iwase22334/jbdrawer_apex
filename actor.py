import numpy as np
import queue
import argparse
import yaml
import zmq
import _pickle as pickle

import torch
import torch.multiprocessing as mp

from environment import Environment
from batch_store import BatchStore
from discriminator import Discriminator
from model import DuelingDQN
from torch.utils.tensorboard import SummaryWriter

import data_generator


def exploration(args, config, epsilon, actor_id, n_actors, batch_queue, param_queue):
    writer = SummaryWriter(log_dir=config["tb_log_path"])

    D = Discriminator().to('cpu')
    D.load_state_dict(torch.load(config["out_path"] + 'D.pth'))
    D.eval()

    # Init meta variables
    batch_store = BatchStore(actor_id, config["n_step"], gamma=config["discount_factor"])

    model = DuelingDQN()
    param = param_queue.get(block=True)
    model.load_state_dict(param)
    param = None
    print("model loaded")

    sequence = 0

    while True:
        for subject_num, (subject, _) in enumerate(data_generator.data_generator(1, False)):
            env = Environment(D, subject, config["image_size"], config["stroke_length"])
            state = env.first_state()

            done = 0
            while done == 0:
                print(subject_num, sequence)

                # Decide action
                qs = model(torch.stack([state])).detach().squeeze().numpy()
                if np.random.rand() <= epsilon:
                    action = torch.randint(low=0, high=Environment.N_WORD, size=(1,)).item()
                else:
                    action = qs.argmax()

                # Play action
                next_state, reward, done = env.step(action)
                batch_store.add(state, action, reward, done, qs)

                state = next_state
                sequence += 1

                if sequence % config["update_interval"] == 0:
                    print("try to update parameter")
                    try:
                        param = param_queue.get(block=True, timeout=None)
                        model.load_state_dict(param)
                        print("Parameter updated ..")
                    except queue.Empty:
                        pass

                if len(batch_store) == config["batch_send_interval"]:
                    print("Sending Batch..")
                    batch, prios = batch_store.make_batch()
                    data = pickle.dumps((batch, prios))
                    batch, prios = None, None
                    batch_store.reset()
                    batch_queue.put(data)

            # update epsilon
            if actor_id == 0 and epsilon > config["epsilon_min"]:
                epsilon *= config["epsilon_decay"]
                writer.add_scalar('meta/epsilon', epsilon, sequence)


def connect_param_socket(ctx, param_socket, learner_ip, actor_id):
    print("connecting to learner")

    socket = ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{learner_ip}:52002")
    socket.send(pickle.dumps((actor_id, 1)))
    socket.recv()
    param_socket.connect(f"tcp://{learner_ip}:52001")
    socket.send(pickle.dumps((actor_id, 2)))
    socket.recv()
    print("Successfully connected to learner!")
    socket.close()


def recv_param(learner_ip, actor_id, param_queue):
    ctx = zmq.Context()
    param_socket = ctx.socket(zmq.SUB)
    param_socket.setsockopt(zmq.SUBSCRIBE, b'')
    param_socket.setsockopt(zmq.CONFLATE, 1)
    connect_param_socket(ctx, param_socket, learner_ip, actor_id)
    while True:
        data = param_socket.recv(copy=False)
        param = pickle.loads(data)

        if param_queue.full():
            try:
                param_queue.get_nowait()
            except queue.Empty:
                pass

        param_queue.put(param)


def send_batch(replay_ip, actor_id, batch_queue):
    ctx = zmq.Context.instance()
    batch_socket = ctx.socket(zmq.DEALER)
    batch_socket.setsockopt(zmq.IDENTITY, pickle.dumps(f'actor-{actor_id}'))
    batch_socket.connect(f"tcp://{replay_ip}:51001")

    while True:
        data = batch_queue.get(block=True)
        batch_socket.send(data, copy=False)
        batch_socket.recv()


def parse_arg():
    parser = argparse.ArgumentParser(description='Ape-X')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--id', type=int)
    parser.add_argument('--n_actors', type=int)
    parser.add_argument('--replay_ip', type=str)
    parser.add_argument('--learner_ip', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    actor_id = args.id
    n_actors = args.n_actors
    if actor_id == 0:
        epsilon = 1.0
    else:
        epsilon = config["eps_base"] ** (1 + actor_id / (n_actors - 1) * config["eps_alpha"])

    print("actor_id, n_actors, epsilon", actor_id, n_actors, epsilon)

    replay_ip = args.replay_ip
    learner_ip = args.learner_ip

    param_queue = mp.Queue(maxsize=1)
    batch_queue = mp.Queue(maxsize=3)

    procs = [
        mp.Process(target=exploration, args=(args, config, epsilon, actor_id, n_actors, batch_queue, param_queue)),
        mp.Process(target=send_batch, args=(replay_ip, actor_id, batch_queue,)),
        mp.Process(target=recv_param, args=(learner_ip, actor_id, param_queue)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
