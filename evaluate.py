import numpy as np
import queue
import argparse
import yaml
import zmq
import _pickle as pickle

import torch
import torch.multiprocessing as mp
import torchvision

from environment import Environment
from batch_store import BatchStore
from discriminator import Discriminator
from model import DuelingDQN
from torch.utils.tensorboard import SummaryWriter

import data_generator


def exploration(args, config, batch_queue, param_queue):
    writer = SummaryWriter(log_dir=config["tb_log_path"])

    D = Discriminator().to('cpu')
    D.load_state_dict(torch.load(config["out_path"] + 'D.pth'))
    D.eval()

    # Init meta variables
    batch_store = BatchStore("-1", config["n_step"], gamma=config["discount_factor"])

    model = DuelingDQN()
    param = param_queue.get(block=True)
    model.load_state_dict(param)
    param = None
    print("model loaded")

    eval_step = 0
    while True:
        reward_hist = []

        for subject_num, (subject, _) in enumerate(data_generator.data_generator(1, False)):
            print("eval_step", eval_step, subject_num)
            total_reward = torch.zeros((1, 1))
            env = Environment(D, subject, config["image_size"], config["stroke_length"])
            state = env.first_state()

            done = 0
            while done == 0:
                # Decide action
                qs = model(torch.stack([state])).detach().squeeze().numpy()
                action = qs.argmax()

                # Play action
                next_state, reward, done = env.step(action)
                batch_store.add(state, action, reward, done, qs)

                state = next_state

                total_reward += reward

                if len(batch_store) == config["batch_send_interval"]:
                    print("Sending Batch..")
                    batch, prios = batch_store.make_batch()
                    data = pickle.dumps((batch, prios))
                    batch, prios = None, None
                    batch_store.reset()
                    batch_queue.put(data)

            reward_hist.append(total_reward.item())

            timg = env.get_subject_img()
            img = env.get_img()
            gtimg = torchvision.utils.make_grid(timg, nrow=1).to('cpu')
            gimg = torchvision.utils.make_grid(img, nrow=1).to('cpu')
            grid_image = torch.cat((gtimg, gimg), dim=2).to('cpu')
            writer.add_image(f'image/ts_{subject_num}', grid_image, eval_step, dataformats='CHW')

            eval_step += 1
            writer.add_scalar('reward/DQN', np.mean(reward_hist), eval_step)

        param = param_queue.get(block=True, timeout=None)
        model.load_state_dict(param)


def connect_param_socket(ctx, param_socket, learner_ip):
    print("connecting to learner")

    socket = ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{learner_ip}:52002")
    socket.send(pickle.dumps((-1, 1)))
    socket.recv()
    param_socket.connect(f"tcp://{learner_ip}:52001")
    socket.send(pickle.dumps((-1, 2)))
    socket.recv()
    print("Successfully connected to learner!")
    socket.close()


def recv_param(learner_ip, param_queue):
    ctx = zmq.Context()
    param_socket = ctx.socket(zmq.SUB)
    param_socket.setsockopt(zmq.SUBSCRIBE, b'')
    param_socket.setsockopt(zmq.CONFLATE, 1)
    connect_param_socket(ctx, param_socket, learner_ip)
    while True:
        data = param_socket.recv(copy=False)
        param = pickle.loads(data)

        if param_queue.full():
            try:
                param_queue.get_nowait()
            except queue.Empty:
                pass

        param_queue.put(param)


def send_batch(replay_ip, batch_queue):
    ctx = zmq.Context.instance()
    batch_socket = ctx.socket(zmq.DEALER)
    batch_socket.setsockopt(zmq.IDENTITY, pickle.dumps('actor--1'))
    batch_socket.connect(f"tcp://{replay_ip}:51001")

    while True:
        data = batch_queue.get(block=True)
        batch_socket.send(data, copy=False)
        batch_socket.recv()


def parse_arg():
    parser = argparse.ArgumentParser(description='Ape-X')
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--replay_ip', type=str)
    parser.add_argument('--learner_ip', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_arg()
    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)

    replay_ip = args.replay_ip
    learner_ip = args.learner_ip

    param_queue = mp.Queue(maxsize=1)
    batch_queue = mp.Queue(maxsize=3)

    procs = [
        mp.Process(target=exploration, args=(args, config, batch_queue, param_queue)),
        mp.Process(target=send_batch, args=(replay_ip, batch_queue,)),
        mp.Process(target=recv_param, args=(learner_ip, param_queue)),
    ]

    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == '__main__':
    main()
