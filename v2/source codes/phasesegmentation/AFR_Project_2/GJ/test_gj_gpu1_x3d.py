import torch
from dataloaders_PJ import PJVideoDataset
from torch.utils.data import Dataset, DataLoader
from data_augmentation import (
    transform_train,
    transform_val,
)
from models import CNN_3D
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    f1_score,
    classification_report,
    accuracy_score,
    confusion_matrix,
    # plot_confusion_matrix,
)
import os
# Huu Phong Nguyen
# os.environ["HTTPS_PROXY"] = "http://proxy.swmed.edu:3128"

test_data_dir = "D:/Research/GJ DL/images"
PATH = "./runs/afr_x3d_a_0_15_v1_2Tool.pth"


positional_encoding = True
num_workers = 8
batch_size = 1
device = torch.device("cuda:1")

num_classes = 7

model = CNN_3D(num_classes=num_classes)
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(device)

model.load_state_dict(torch.load(PATH))
model.eval()

test_dataset = PJVideoDataset(
    data_dir=test_data_dir,
    transform=transform_val,
    positional_encoding=True,
    train=False,
    split="test",
    clip_length=16,
)
dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

dataset_length = len(test_dataset)


def run():
    # torch.multiprocessing.freeze_support()

    running_corrects = 0
    y_true = []
    y_pred = []

    epoch_acc = 0
    epoch_f1 = 0

    with tqdm(total=dataset_length) as epoch_pbar:
        epoch_pbar.set_description(f"validating")
        with torch.no_grad():
            for batch, (image, target, frame_num, path) in enumerate(dataloader):

                frame_num = frame_num.to(device)
                inputs = image.to(device)

                labels = target.to(device)

                outputs = model(inputs, frame_num)[0]

                _, preds = torch.max(outputs, 1)

                with open("afr_x3d_a_0_15_v1_2Tool.txt", "a") as f:
                    p = str(path).split(".")[0].split("/")[-1]
                    f.write(
                        f"{p}, {labels.cpu().data.numpy()}, {preds.cpu().data.numpy()}\n"
                    )

                running_corrects += torch.sum(preds == labels.data)

                y_true = np.concatenate((labels.cpu().data.numpy(), y_true))
                y_pred = np.concatenate((preds.cpu().data.numpy(), y_pred))

                epoch_pbar.update(inputs.shape[0])
                epoch_pbar.set_postfix_str(f'acc={running_corrects.double()/epoch_pbar.n:.4f}')

    epoch_acc = running_corrects.double() / dataset_length
    epoch_f1 = f1_score(y_pred=y_pred, y_true=y_true, average="macro")

    print(f"Acc: {epoch_acc} F1: {epoch_f1}")

    print(classification_report(y_true=y_true, y_pred=y_pred))
    print(accuracy_score(y_true=y_true, y_pred=y_pred))


if __name__ == "__main__":
    run()
