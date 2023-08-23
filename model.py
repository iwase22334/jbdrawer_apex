from torch import nn
from environment import Environment

model_parameter = {
    "d_image": 64,
    "d_canvas": 64 * 2,
    "conv_channel1": 16,
    "conv_channel2": 32,
    "conv_channel3": 64,
    "d_hidden": 512,
    "d_dict": Environment.N_WORD
}


class DuelingDQN(nn.Module):
    def __init__(self):
        super().__init__()
        param = model_parameter

        hidden = param["d_hidden"]
        image_size = param["d_canvas"]
        channel1 = param["conv_channel1"]
        channel2 = param["conv_channel2"]
        channel3 = param["conv_channel3"]
        self.output_size = param["d_dict"]

        # output_size = (input_size - kernel_size + 2 × padding) ÷ stride + 1
        d_in1 = (image_size - 3 + 2 * 1) // 1 + 1

        self.seq = nn.Sequential(
            nn.Conv2d(2, channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(channel1, channel2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(channel2, channel3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Flatten(),

            nn.Linear(channel3 * d_in1 * d_in1, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.value_stream_layer = nn.Sequential(nn.Linear(hidden, hidden),
                                                nn.ReLU())
        self.advantage_stream_layer = nn.Sequential(nn.Linear(hidden, hidden),
                                                    nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, self.output_size)


    def forward(self, state):
        # MainUnit
        x = self.seq(state)

        value = self.value(self.value_stream_layer(x))
        advantage = self.advantage(self.advantage_stream_layer(x))
        action_value = value + (advantage - (1 / self.output_size) * advantage.sum() )

        return action_value
