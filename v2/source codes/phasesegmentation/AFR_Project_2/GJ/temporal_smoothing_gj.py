import numpy as np
import os
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    classification_report,
    # plot_roc_curve,
    confusion_matrix,
    jaccard_score,
)
import math


with open("./GJ_ensemble_x3d_0_1_v1_4Tool.txt", "r") as f:
    lines_ = f.readlines()
    # lines.sort()

lines = []
for line in lines_:
    frame_num = line.split("Frame")[1].split(",")[0]
    lines.append(
        f'{line.split("Frame")[0]}Frame{str(frame_num).zfill(6)}, {line.split(", ")[1]}, {line.split(", ")[2]}'
    )
    # print(lines)

lines.sort()
y_true = []
y_pred = []

for line in lines:
    # print(line)
    y_true.append(int(line.split(",")[1].split("[")[1].split("]")[0]))
    y_pred.append(int(line.split(",")[2].split("[")[1].split("]")[0]))

accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")

print(accuracy, f1)

window_size = 31
l = math.floor(window_size / 2)

moving_average = []


def most_frequent(List):
    return max(set(List), key=List.count)


i = l
while i < len(y_pred) - l:
    window = y_pred[i - l : i + l]

    # window_average = round(sum(window) / window_size, 2)
    window_mode = most_frequent(window)
    moving_average.append(round(window_mode))

    i += 1

y_pred1 = y_pred[:l] + moving_average + y_pred[-l:]

accuracy = accuracy_score(y_true=y_true, y_pred=y_pred1)
f1 = f1_score(y_true=y_true, y_pred=y_pred1, average="macro")
jaccard = jaccard_score(y_true=y_true, y_pred=y_pred1, average=None)


print(accuracy, f1, jaccard)

print(classification_report(y_true=y_true, y_pred=y_pred1))
cf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred1, normalize="true")

import seaborn as sns
import matplotlib.pyplot as plt


ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues")

ax.set_xlabel("\nPredicted Values")
ax.set_ylabel("Actual Values ")

# ['Other tasks/events',
# 'Superior mattress stitch',
# 'Middle mattress stitch',
# 'Inferior mattress stitch',
# 'Grasp superior stitch w/ 3rd arm',
# 'Confirm patency of pancreatic duct',
# 'Make enterotomy',
# 'Posterior duct-to-mucosa stitch',
# 'Anterior duct-to-mucosa stitch',
# 'Place stent',
# 'Superior buttress stitch',
# 'Middle buttress stitch',
# 'Inferior buttress stitch']


# ax.xaxis.set_ticklabels(['False','True'])
# ax.yaxis.set_ticklabels(['False','True'])

plt.show()
