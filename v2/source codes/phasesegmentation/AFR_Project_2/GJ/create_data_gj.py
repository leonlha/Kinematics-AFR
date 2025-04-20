import os
import random
import glob

import numpy as np
import pandas as pd
import cv2
from joblib import delayed, Parallel

ANNOTATION_PATH = "D:/Research/GJ DL/4 - New Annotations"
VIDEOS_PATH = "D:/Research/GJ DL/5- New Videos"
IMAGES_PATH = "D:/Research/GJ DL/images"

num_videos = 42
num_steps = 7

steps_dic = {
    "1.1 Stay suture": 1,
    "1.2 Inner running suture": 2,
    "1.3 Enterotomy": 3, 
    "2.2 Inner running suture": 4,
    "3.1 Inner Layer of Connell": 5,
    "4.1 Outer layer of Connell": 6,
    "Nothing": 0
}

NUM_FRAME_PER_STEP = 250 #200  # CLACULATED BASED ON THE MEDIAN


def convert_time_to_sec(time):
    if time != "0":
        h, m, s = str(time).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return int(0)


def convert_sec_to_time(sec):
    m, s = divmod(sec, 60)
    return f"0:{int(m)}:{int(s)}"


# # Phong
# # list all annotated videos and shuffle them
# videos = []
# for v in glob.glob(f"{ANNOTATION_PATH}/*/"):
#     videos.append(v.split("\\")[-2])

# # random.shuffle(videos)

# # calculate statistics, find the median for each step
# steps_durations = np.zeros((num_videos, num_steps))

# for index, video in enumerate(videos):
#     with open(f"{ANNOTATION_PATH}/{video}/GJ.txt", "r") as f:
#         lines = f.readlines()
#         video_length = convert_time_to_sec(
#             lines[-1].split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
#         )

#         for step in steps_dic:
#             if step != "Nothing":
#                 step_duration = 0
#                 for line in lines:
#                     if step == line.split(" : ")[0]:
#                         b = convert_time_to_sec(
#                             line.split(" : ")[1].split("(")[1].split(",")[0]
#                         )
#                         e = convert_time_to_sec(
#                             line.split(" : ")[1]
#                             .split("(")[1]
#                             .split(",")[1]
#                             .split(")")[0]
#                         )

#                         step_duration += int(e) - int(b)

#                 steps_durations[index, int(steps_dic[step])] = step_duration
        
#         # print(f"{ANNOTATION_PATH}/{video}/GJ.txt", ":", steps_durations[index])
#         sum_steps_duration = np.sum(steps_durations[index, :])
#         steps_durations[index, 0] = video_length - sum_steps_duration
#         # print(f"{ANNOTATION_PATH}/{video}/GJ.txt", ":", steps_durations[index])

# print(steps_durations)

# print(np.median(steps_durations, axis=0))
# print(np.count_nonzero(steps_durations, axis=0))

# median of all steps in all videos = 259.5



# create a csv file with image name, label, video length, fps

def create_txt_train(video, annotated):

    video_name = video.split("\\")[-1]

    if video_name in annotated:
        print(video_name)
        cap = cv2.VideoCapture(video)
        frame_rate = round(cap.get(cv2.CAP_PROP_FPS))

        with open(f"{ANNOTATION_PATH}/{video_name}/GJ.txt", "r") as f:
            lines = f.readlines()
            video_length = convert_time_to_sec(
                lines[-1].split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
            )

            for step in steps_dic:
                if step != "Nothing":

                    # save the indices (time or frame_num*framne_rate) for the step
                    step_indices = []
                    for line in lines:
                        if step == line.split(" : ")[0]:

                            b = convert_time_to_sec(
                                line.split(" : ")[1].split("(")[1].split(",")[0]
                            )
                            e = convert_time_to_sec(
                                line.split(" : ")[1]
                                .split("(")[1]
                                .split(",")[1]
                                .split(")")[0]
                            )

                            for i in range(b, e):
                                step_indices.append(i)

                    if len(step_indices) > NUM_FRAME_PER_STEP:
                        step_indices_sample = random.sample(
                            step_indices, NUM_FRAME_PER_STEP
                        )
                        for i in range(NUM_FRAME_PER_STEP):
                            with open("./train.txt", "a") as train_file:
                                train_file.write(
                                    f'{video_name.split(".")[0]}Frame{str(step_indices_sample[i]*frame_rate).zfill(5)}.jpg---{steps_dic[step]}---{video_length}---{frame_rate}\n'
                                )

                    else:
                        if len(step_indices) != 0:
                            s, r = divmod(NUM_FRAME_PER_STEP, len(step_indices))
                            for i in range(len(step_indices)):
                                with open("./train.txt", "a") as train_file:
                                    train_file.write(
                                        f'{video_name.split(".")[0]}Frame{str(step_indices[i]*frame_rate).zfill(5)}.jpg---{steps_dic[step]}---{video_length}---{frame_rate}\n'
                                    )
                            for _ in range(s - 1):
                                for i in range(len(step_indices)):
                                    with open("./train.txt", "a") as train_file:
                                        train_file.write(
                                            f'{video_name.split(".")[0]}Frame{str((step_indices[i] *frame_rate) + random.randint(1,5) * (int(frame_rate / 6))).zfill(5)}.jpg---{steps_dic[step]}---{video_length}---{frame_rate}\n'
                                        )
                            step_indices_sample = random.sample(step_indices, r)
                            for i in range(r):
                                with open("./train.txt", "a") as train_file:
                                    train_file.write(
                                        f'{video_name.split(".")[0]}Frame{str(step_indices_sample[i]*frame_rate).zfill(5)}.jpg---{steps_dic[step]}---{video_length}---{frame_rate}\n'
                                    )

                else:
                    allstep_indices = []

                    for line in lines:
                        b = convert_time_to_sec(
                            line.split(" : ")[1].split("(")[1].split(",")[0]
                        )
                        e = convert_time_to_sec(
                            line.split(" : ")[1]
                            .split("(")[1]
                            .split(",")[1]
                            .split(")")[0]
                        )
                        for i in range(b, e):
                            allstep_indices.append(i)

                    nothing_indices_len = video_length - len(allstep_indices)

                    if nothing_indices_len > NUM_FRAME_PER_STEP:
                        i = 0
                        while i < NUM_FRAME_PER_STEP:
                            rnd_num = random.randint(1, video_length)
                            if not rnd_num in allstep_indices:
                                with open("./train.txt", "a") as train_file:
                                    train_file.write(
                                        f'{video_name.split(".")[0]}Frame{str(rnd_num*frame_rate).zfill(5)}.jpg---0---{video_length}---{frame_rate}\n'
                                    )
                                i += 1
                    else:

                        s, r = divmod(NUM_FRAME_PER_STEP, nothing_indices_len)

                        for _ in range(s):
                            i = 0
                            while i < nothing_indices_len:
                                rnd_num = random.randint(1, video_length)
                                if not rnd_num in allstep_indices:
                                    with open("./train.txt", "a") as train_file:
                                        train_file.write(
                                            f'{video_name.split(".")[0]}Frame{str(rnd_num*frame_rate).zfill(5)}.jpg---0---{video_length}---{frame_rate}\n'
                                        )
                                    i += 1
                        i = 0
                        while i < r:
                            rnd_num = random.randint(1, video_length)
                            if not rnd_num in allstep_indices:
                                with open("./train.txt", "a") as train_file:
                                    train_file.write(
                                        f'{video_name.split(".")[0]}Frame{str(rnd_num*frame_rate).zfill(5)}.jpg---0---{video_length}---{frame_rate}\n'
                                    )
                                i += 1


def create_txt_val(video, annotated, val: bool):
    video_name = video.split("\\")[-1]

    if val:
        out = "val"
    else:
        out = "test"

    if video_name in annotated:
        print(video_name)
        cap = cv2.VideoCapture(video)
        frame_rate = round(cap.get(cv2.CAP_PROP_FPS))

        with open(f"{ANNOTATION_PATH}/{video_name}/GJ.txt", "r") as f:
            lines = f.readlines()
            video_length = convert_time_to_sec(
                lines[-1].split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
            )

            for step in steps_dic:
                if step != "Nothing":

                    for line in lines:
                        if step == line.split(" : ")[0]:

                            b = convert_time_to_sec(
                                line.split(" : ")[1].split("(")[1].split(",")[0]
                            )
                            e = convert_time_to_sec(
                                line.split(" : ")[1]
                                .split("(")[1]
                                .split(",")[1]
                                .split(")")[0]
                            )

                            for i in range(b, e):
                                with open(f"./{out}.txt", "a") as file:
                                    file.write(
                                        f'{video_name.split(".")[0]}Frame{str(i*frame_rate).zfill(5)}.jpg---{steps_dic[step]}---{video_length}---{frame_rate}\n'
                                    )
                else:
                    allstep_indices = []

                    for line in lines:
                        b = convert_time_to_sec(
                            line.split(" : ")[1].split("(")[1].split(",")[0]
                        )
                        e = convert_time_to_sec(
                            line.split(" : ")[1]
                            .split("(")[1]
                            .split(",")[1]
                            .split(")")[0]
                        )
                        for i in range(b, e):
                            allstep_indices.append(i)

                    for i in range(video_length):
                        if not i in allstep_indices:
                            with open(f"./{out}.txt", "a") as file:
                                file.write(
                                    f'{video_name.split(".")[0]}Frame{str(i*frame_rate).zfill(5)}.jpg---{steps_dic[step]}---{video_length}---{frame_rate}\n'
                                )


if __name__ == "__main__":

    # list all annotated videos and shuffle them
    annotated_videos = []
    for v in glob.glob(f"{ANNOTATION_PATH}/*/"):
        annotated_videos.append(v.split("\\")[-2])

    print(len(annotated_videos))
    print(annotated_videos)

    random.shuffle(annotated_videos)
    train_videos = annotated_videos[:30]
    val_videos = annotated_videos[30:42]
    test_videos = annotated_videos[42:]
    print(len(train_videos), len(val_videos), len(test_videos))

    videos_mp4 = glob.glob(f"{VIDEOS_PATH}/*/*.mp4", recursive=True)
    videos_mov = glob.glob(f"{VIDEOS_PATH}/*/*.mov", recursive=True)

    videos = videos_mov + videos_mp4
    print(len(videos))

    for video in videos:
        video_name = video.split("\\")[-1]
        # print(video_name)

        if video_name in train_videos:
            create_txt_train(video=video, annotated=train_videos)
        elif video_name in val_videos:
            create_txt_val(video=video, annotated=val_videos, val=True)
        else:
            create_txt_val(video=video, annotated=test_videos, val=False)

    print(len(train_videos), len(val_videos), len(test_videos))

    with open("train.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open("train.txt", "w") as f:
        f.writelines(lines)

    with open("val.txt", "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open("val.txt", "w") as f:
        f.writelines(lines)

    # Parallel(n_jobs=60)(delayed(create_txt_train)(vid) for vid in videos)
