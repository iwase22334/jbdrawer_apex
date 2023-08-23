from tqdm import tqdm
import numpy as np
import os

import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import data_generator


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.inlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        self.inlayer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        self.main = nn.Sequential(
            nn.Conv2d(2 * 32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(62 * 62, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, image):
        x1 = self.inlayer1(x)
        x2 = self.inlayer2(image)
        inp = torch.cat((x1, x2), dim=1)

        return self.main(inp)


# Train Discriminator
def _D_train_step(D, D_optim, x, device):
    t_data, f_data = torch.split(x, [1, 1], dim=0)
    t_data, f_data = t_data.to(device), f_data.to(device)

    y_pred = D(t_data, t_data)
    y_real = torch.full_like(y_pred, 1, device=device)
    loss_real = nn.BCELoss()(y_pred, y_real)

    D.zero_grad()
    loss_real.backward()
    D_optim.step()

    y_pred = D(t_data, f_data)
    y_fake = torch.full_like(y_pred, 0, device=device)
    loss_fake = nn.BCELoss()(y_pred, y_fake)

    D.zero_grad()
    loss_fake.backward()
    D_optim.step()

    loss = loss_real + loss_fake
    return float(loss)


# Train D
def D_train(learning_rate, num_epoch, writer, device):
    D = Discriminator().to(device)
    D_optim = optim.Adam(D.parameters(), lr=learning_rate)

    for epoch in tqdm(range(num_epoch)):
        pre_D_losses = []
        for x, _ in data_generator.data_generator(2, True):
            noise = torch.randn_like(x) * 0.2
            x = x + noise
            x = torch.clamp(x, max=1.0, min=0.0).squeeze(0)

            loss_d = _D_train_step(D, D_optim, x.to(device), device)
            pre_D_losses.append(loss_d)
            print("D:", float(loss_d))

        writer.add_scalar('Loss/D_train_step', np.mean(pre_D_losses), epoch)

    D.eval()
    del D_optim
    torch.cuda.empty_cache()

    return D


def parse_arg():
    parser = argparse.ArgumentParser(description='Ape-X')
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arg()

    with open(args.config_file, 'r') as file:
        param = yaml.safe_load(file)

    os.makedirs(param['out_path'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = param["pre_learning_rate"]
    num_epoch = param["pre_num_epoch"]
    writer = SummaryWriter(log_dir=param["tb_log_path"])

    model = D_train(learning_rate, num_epoch, writer, device)
    torch.save(model.state_dict(), param['out_path'] + 'D.pth')
