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

# os.environ["HTTPS_PROXY"] = "http://proxy.swmed.edu:3128"

test_data_dir = "D:/Research/GJ DL/images"
PATH = "./runs/afr_x3d_a_0_15_v1_2Tool.pth"

test_data_dir2 = "D:/Research/GJ DL/opticalflows"
PATH2 = "./runs/afr_opticalflow_gj_x3d_a_0_15_v1_2Tool.pth"

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

#model2
model2 = CNN_3D(num_classes=num_classes)
# model = torch.nn.DataParallel(model, device_ids=[0, 1])
model2.to(device)
model2.load_state_dict(torch.load(PATH2))
model2.eval()

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

test_dataset2 = PJVideoDataset(
    data_dir=test_data_dir2,
    transform=transform_val,
    positional_encoding=True,
    train=False,
    split="test",
    clip_length=16,
)
dataloader2 = DataLoader(
    test_dataset2, batch_size=batch_size, shuffle=False, num_workers=num_workers
)


dataset_length = len(test_dataset)
dataset_length2 = len(test_dataset2)

print(dataset_length, dataset_length2)

# def run():
#     # torch.multiprocessing.freeze_support()

#     running_corrects = 0
#     y_true = []
#     y_pred = []

#     epoch_acc = 0
#     epoch_f1 = 0

#     with tqdm(total=dataset_length) as epoch_pbar:
#         epoch_pbar.set_description(f"validating")
#         with torch.no_grad():
#             for batch, (image, target, frame_num, path) in enumerate(dataloader):

#                 frame_num = frame_num.to(device)
#                 inputs = image.to(device)

#                 labels = target.to(device)

#                 outputs = model(inputs, frame_num)[0]

#                 _, preds = torch.max(outputs, 1)

#                 with open("GJ_results_x3d.txt", "a") as f:
#                     p = str(path).split(".")[0].split("/")[-1]
#                     f.write(
#                         f"{p}, {labels.cpu().data.numpy()}, {preds.cpu().data.numpy()}\n"
#                     )

#                 running_corrects += torch.sum(preds == labels.data)

#                 y_true = np.concatenate((labels.cpu().data.numpy(), y_true))
#                 y_pred = np.concatenate((preds.cpu().data.numpy(), y_pred))

#                 epoch_pbar.update(inputs.shape[0])
#                 epoch_pbar.set_postfix_str(f'acc={running_corrects.double()/epoch_pbar.n:.4f}')

#     epoch_acc = running_corrects.double() / dataset_length
#     epoch_f1 = f1_score(y_pred=y_pred, y_true=y_true, average="macro")

#     print(f"Acc: {epoch_acc} F1: {epoch_f1}")

#     print(classification_report(y_true=y_true, y_pred=y_pred))
#     print(accuracy_score(y_true=y_true, y_pred=y_pred))


# if __name__ == "__main__":
#     run()

def run():
    # torch.multiprocessing.freeze_support()

    running_corrects = 0
    y_true = []
    y_pred = []
    results_to_write = [] 
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

                results_to_write.append((path, labels.cpu().data.numpy(), outputs.cpu().data.numpy()))  # Append outputs

                _, preds = torch.max(outputs, 1)

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


    ########################## Model 2 #####################
    running_corrects2 = 0
    y_true2 = []
    y_pred2 = []
    results_to_write2 = [] 
    epoch_acc2 = 0
    epoch_f12 = 0

    with tqdm(total=dataset_length2) as epoch_pbar:
        epoch_pbar.set_description(f"validating2")
        with torch.no_grad():
            for batch, (image, target, frame_num, path) in enumerate(dataloader2):

                frame_num = frame_num.to(device)
                inputs = image.to(device)

                labels = target.to(device)

                outputs2 = model2(inputs, frame_num)[0]

                results_to_write2.append((path, labels.cpu().data.numpy(), outputs2.cpu().data.numpy()))  # Append outputs

                _, preds2 = torch.max(outputs2, 1)

                running_corrects2 += torch.sum(preds2 == labels.data)

                y_true2 = np.concatenate((labels.cpu().data.numpy(), y_true2))
                y_pred2 = np.concatenate((preds2.cpu().data.numpy(), y_pred2))

                epoch_pbar.update(inputs.shape[0])
                epoch_pbar.set_postfix_str(f'acc={running_corrects2.double()/epoch_pbar.n:.4f}')

    epoch_acc2 = running_corrects2.double() / dataset_length2
    epoch_f12 = f1_score(y_pred=y_pred2, y_true=y_true2, average="macro")

    print(f"Acc: {epoch_acc2} F1: {epoch_f12}")

    print(classification_report(y_true=y_true2, y_pred=y_pred2))
    print(accuracy_score(y_true=y_true2, y_pred=y_pred2))

    # with open("GJ_results_ensemble_x3d.txt", "a") as f:
    #     for path, true_labels, output in results_to_write:
    #         p = str(path).split(".")[0].split("/")[-1]
    #         preds = np.argmax(output, axis=1)
    #         f.write(f"{p}, {true_labels}, {preds}\n")

    # Ensemble results
    ensemble_results = []
    for (path, true_labels, output), (_, _, output2) in zip(results_to_write, results_to_write2):
        ensemble_output = (output + output2) / 2  #
        ensemble_results.append((path, true_labels, ensemble_output))

    # Compute epoch_acc and epoch_f1
    y_true = np.concatenate([true_labels for _, true_labels, _ in ensemble_results])
    y_pred = np.concatenate([np.argmax(output, axis=1) for _, _, output in ensemble_results])
    epoch_acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    epoch_f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    # Print classification report and accuracy score
    print(classification_report(y_true=y_true, y_pred=y_pred))
    print(f"Epoch Accuracy: {epoch_acc}")
    print(f"Epoch F1 Score: {epoch_f1}")

    with open("GJ_ensemble_x3d_0_15_v1_2Tool.txt", "a") as f:
        for path, true_labels, ensemble_output in ensemble_results:
            p = str(path).split(".")[0].split("/")[-1]
            preds = np.argmax(ensemble_output, axis=1)
            f.write(f"{p}, {true_labels}, {preds}\n")

if __name__ == "__main__":
    run()
