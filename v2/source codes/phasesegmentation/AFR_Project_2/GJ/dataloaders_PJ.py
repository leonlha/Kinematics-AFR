from cv2 import transform
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
import PIL
from PIL import Image
import io
import os
import math
import numpy as np
import matplotlib.pyplot as plt


# def positional_encoding(pos, n_dim, max_length: int = 10000):
#     # batch = len(pos)
#     pe = torch.zeros(n_dim)
#     for i in range(0, n_dim, 2):
#         pe[i] = math.sin(pos / (max_length ** ((2 * i) / n_dim)))
#         pe[i + 1] = math.cos(pos / (max_length ** ((2 * (i + 1)) / n_dim)))
#     return pe  # .to("cuda:0")
def positional_encoding(pos, n_dim, max_length: int = 10000):
    pe = torch.zeros(n_dim)
    for i in range(0, n_dim, 1):
        if i % 2 == 0:  # Check if i is even
            pe[i] = math.sin(pos / (max_length ** ((2 * i) / n_dim)))
        else:
            pe[i] = math.cos(pos / (max_length ** ((2 * i) / n_dim)))
    return pe


class PJImageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        train=True,
        positional_encoding=False,
        split="train",
    ):

        self.data_dir = data_dir
        self.transform = transform
        self.positional_encoding = positional_encoding
        self.train = train
        self.split = split
        self.label_path = os.path.join(f"{self.split}.txt")
        with open(self.label_path, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx]

        img_path = os.path.join(
            self.data_dir,
            line.split("---")[0].split("Frame")[0],
            line.split("---")[0],
        )

        image = PIL.Image.open(img_path)
        label = int(line.split("---")[1])
        video_length = int(line.split("---")[2])
        fps = int(line.split("---")[3])
        frame_num = img_path.split("/")[-1].split(".")[0].split("Frame")[1]

        # #return  {'images': self.transform(images[5]), 'label':labels[5], 'path': img_paths}
        pe = torch.zeros(1536)
        # divide pos by frame rate - add an argument for video length
        if self.positional_encoding:
            pe = positional_encoding(
                pos=int(frame_num / fps),
                n_dim=1536,
                max_length=video_length,
            )

        return self.transform(image), label, pe, img_path  # [5]


class PJVideoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        train=True,
        positional_encoding=False,
        split="train",
        clip_length=2,
        flow=False,
    ):

        self.data_dir = data_dir
        self.transform = transform
        self.positional_encoding = positional_encoding
        self.train = train
        self.split = split
        self.label_path = os.path.join(f"{self.split}.txt")
        with open(self.label_path, "r") as f:
            self.lines = f.readlines()

        self.clip_length = clip_length
        self.flow = flow

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx]

        img_path = os.path.join(
            self.data_dir,
            line.split("---")[0].split("Frame")[0],
            line.split("---")[0],
        )

        # image = PIL.Image.open(img_path)
        label = int(line.split("---")[1])
        video_length = int(line.split("---")[2])
        fps = int(line.split("---")[3])
        frame_num = img_path.split("/")[-1].split(".")[0].split("Frame")[1]

        images = []
        for i in range(self.clip_length - 1, 0, -1):

            frame_num_ = str(int(frame_num) - int((fps * i) / 6)).zfill(5)
            if int(frame_num_) < 0:
                frame_num_ = "00000"

            if self.flow:
                if not img_path.__contains__("PW06212016JK"):

                    flow_img = np.load(
                        f'{img_path.split("Frame")[0]}Frame{frame_num_}.npy'
                    )
                    additional_channel = np.zeros(
                        (flow_img.shape[0], 1, flow_img.shape[-2], flow_img.shape[-1])
                    )

                    flow_img = np.append(flow_img, additional_channel, axis=-3)

                    images.append(torch.from_numpy(flow_img))
            else:
                images.append(
                    torchvision.io.read_image(
                        f'{img_path.split("Frame")[0]}Frame{frame_num_}.jpg'
                    )
                )

        if self.flow:
            if not img_path.__contains__("PW06212016JK"):
                flow_img = np.load(f'{img_path.split("Frame")[0]}Frame{frame_num}.npy')
                additional_channel = np.zeros(
                    (flow_img.shape[0], 1, flow_img.shape[-2], flow_img.shape[-1])
                )

                flow_img = np.append(flow_img, additional_channel, axis=-3)
                images.append(torch.from_numpy(flow_img))
        else:
            images.append(torchvision.io.read_image(img_path))

        images_tensor = torch.stack(images)  # T, C, H, W

        # images_tensor.to("cuda")

        pe = torch.zeros(2048)  # 7, 2048, 1536
        # divide pos by frame rate - add an argument for video length
        if self.positional_encoding:
            pe = positional_encoding(
                pos=int(int(frame_num) / fps),
                n_dim=2048,#7
                max_length=video_length,
            )

        scripted_transforms = torch.jit.script(self.transform)

        images = scripted_transforms(torch.squeeze(images_tensor))

        return torch.permute(images, (1, 0, 2, 3)), label, pe, img_path


class PJTwoStreamDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        train=True,
        positional_encoding=False,
        split="train",
        clip_length=2,
    ):

        self.data_dir = data_dir
        self.transform = transform
        self.positional_encoding = positional_encoding
        self.train = train
        self.split = split
        self.label_path = os.path.join(f"{self.split}.txt")
        with open(self.label_path, "r") as f:
            self.lines = f.readlines()

        self.clip_length = clip_length

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.lines[idx]

        img_path = os.path.join(
            self.data_dir,
            line.split("---")[0].split("Frame")[0],
            line.split("---")[0],
        )

        image = PIL.Image.open(img_path)
        label = int(line.split("---")[1])
        video_length = int(line.split("---")[2])
        fps = int(line.split("---")[3])
        frame_num = img_path.split("/")[-1].split(".")[0].split("Frame")[1]

        images = []
        for i in range(self.clip_length - 1, 0, -1):

            frame_num_ = str(int(frame_num) - int((fps * i) / 6)).zfill(5)
            if int(frame_num_) < 0:
                frame_num_ = "00000"
            images.append(
                torchvision.io.read_image(
                    f'{img_path.split("Frame")[0]}Frame{frame_num_}.jpg'
                )
            )
        images.append(torchvision.io.read_image(img_path))

        images_tensor = torch.stack(images)  # T, C, H, W

        # images_tensor.to("cuda")

        pe = torch.zeros(2048 + 1536)  # 1536
        # divide pos by frame rate - add an argument for video length
        if self.positional_encoding:
            pe = positional_encoding(
                pos=int(int(frame_num) / fps),
                n_dim=2048 + 1536,
                max_length=video_length,
            )

        scripted_transforms = torch.jit.script(self.transform)

        images = scripted_transforms(images_tensor)

        images = torch.permute(images, (1, 0, 2, 3))

        # print(images[:, 0, :, :].shape)

        return (
            images,
            images[:, 0, :, :],
            label,
            pe,
            img_path,
        )


if __name__ == "__main__":
    from data_augmentation import transform_train, transform_val
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import Compose, Lambda, Resize

    transform1 = torch.nn.Sequential(
        transforms.Resize(
            (300, 300), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomApply(torch.nn.ModuleList([transforms.RandAugment()]), 0.85),
        transforms.ConvertImageDtype(torch.float),
        # transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
        # transforms.RandomErasing(
        #     p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False
        # ),
    )

    dataset = PJVideoDataset(
        data_dir="D:\data\PJ\\images",
        transform=transform1,
        positional_encoding=False,
        train=False,
        split="test",
        clip_length=16,
        flow=True,
    )

    data = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )
    iterator = iter(data)

    # for _ in range(10):

    inputs, label, _, _ = iterator.next()

    # out = transform_val(inputs)

    print(inputs.shape)

    # grid = torchvision.utils.make_grid(torch.squeeze(inputs), nrow=5)

    # img = torchvision.transforms.ToPILImage()(grid)
    # img.show()
    # inputs.cpu().detach().numpy()
    # print(inputs.max())
