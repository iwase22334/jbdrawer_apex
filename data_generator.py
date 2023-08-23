import torch
import torchvision
from torchvision import transforms


def data_generator(bnum=1, shuffle=True):
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    batch_size = bnum
    dataset = torchvision.datasets.ImageFolder("data/images", transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
