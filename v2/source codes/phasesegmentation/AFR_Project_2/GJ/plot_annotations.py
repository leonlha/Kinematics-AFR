import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import glob

height = 0.3
path = "D:/data/PJ/1- Annotations/New Annotations Final"

file_list = glob.glob(f"{path}/*/PJ.txt")
# file_list = file_list + glob.glob(f'{path}/*/*/*.txt')
print(len(file_list))
videos = []

for file in file_list:
    videos.append(str(file.split("\\")[-2]))

print(len(videos))


def convert_time_to_sec(time):
    if time != "0":
        h, m, s = str(time).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return int(0)


def convert_sec_to_time(sec):
    m, s = divmod(sec, 60)
    return f"{int(m)}:{int(s)}"


for video in videos:
    print(video)
    for t in ["task"]:

        task_names = [
            task.split("#")[0].split("\n")[0].rstrip()
            for task in open("PJ_steps.txt", "r").readlines()
        ]
        # print(task_names)

        tasks_dic = dict(zip(task_names, list(range(1, len(task_names) + 1))))
        tasks_dic["nothing"] = 0
        # print(tasks_dic)

        duration = 0
        with open(path + f"/{video}/PJ.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    end = convert_time_to_sec(
                        line.split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
                    )
                except:
                    print("err", video)
                if end > duration:
                    duration = end

        x = [None] * (duration)
        y = [None] * (len(task_names) + 1)
        for i in list(range(duration)):
            x[i] = convert_sec_to_time(i)
        for key, value in tasks_dic.items():
            y[value] = key.rstrip()
        #
        plt.figure(figsize=(30, 15))
        plt.locator_params(axis="x", nbins=50, tight=True)
        # axes = plt.axes()
        plt.grid(True)
        axes = plt.axes()
        # print(axes.get_xticks())
        # axes.set_xticklabels([2,3,4])
        # axes.set_xticklabels(x)
        #
        with open(path + f"/{video}/PJ.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                Y = np.arange(len(task_names) + 1)
                X_left = np.zeros(len(task_names) + 1, dtype=int)
                X_diff = np.zeros(len(task_names) + 1, dtype=int)
                task_name = line.split(" : ")[0].rstrip()
                start = convert_time_to_sec(
                    line.split(" : ")[1].split("(")[1].split(",")[0]
                )
                end = convert_time_to_sec(
                    line.split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
                )
                diff = end - start
                X_left[tasks_dic[task_name]] = start
                X_diff[tasks_dic[task_name]] = diff
                plt.barh(
                    y=Y - height / 2,
                    width=X_diff,
                    left=X_left,
                    height=height,
                    color="blue",
                    label="steps",
                )

        plt.title(f"{video} ")
        ticks = axes.get_xticks()
        x_label = []
        for i in list(range(len(ticks))):
            x_label.append(convert_sec_to_time(ticks[i]))
        plt.xticks(ticks, x_label, rotation="vertical")
        plt.yticks(list(range(len(task_names) + 1)), y)

        # axes.set_xticklabels(x)
        plt.locator_params(axis="x", nbins=100, tight=True)
        # axes.locator_params(axis="x", nbins=50)
        ##plt.locator_params(axis="y", nbins=21)

        plt.savefig(f"D:/plots_new1/{video}.png")
# plt.show()
