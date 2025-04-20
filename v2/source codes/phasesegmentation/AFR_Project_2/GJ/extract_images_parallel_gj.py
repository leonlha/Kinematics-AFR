import os
import glob

from joblib import delayed, Parallel
import cv2
import numpy as np

"""
Extract from the videos at a fixed fps. images are cropped to remove the black margin. 
Videos are trimmed between the minimum time of the first step and ending of the last. 
Only images are saved, not videos
"""

VIDEOS_PATH = "D:/Research/GJ DL/5- New Videos"
IMAGES_PATH = "D:/Research/GJ DL/images"
ANNOTATION_PATH = "D:/Research/GJ DL/4 - New Annotations"


def convert_time_to_sec(time):
    if time != "0":
        h, m, s = str(time).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    else:
        return int(0)


def crop_image(img, tol=20):

    mask = img > tol
    # mask = mask < 240

    if img.ndim == 3:
        mask = mask.all(2)
    m, n = mask.shape
    # return img[np.ix_(mask.any(1), mask.any(0))]
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    # return img[row_start:row_end, col_start:col_end]
    return row_start, row_end, col_start, col_end


def trim_video(video: str):
    # returns the beginning of the first step and the ending of the last step
    video_name = video.split("\\")[-1]

    annotation_file = f"{ANNOTATION_PATH}/{video_name}/GJ.txt"

    duration = 0
    with open(annotation_file, "r") as f:
        lines = f.readlines()
        # for line in lines:
        # if line.startswith('1.1 Anterior Mattress Suture'):
        beginning = convert_time_to_sec(
            lines[0].split(" : ")[1].split("(")[1].split(",")[0]
        )
        end = convert_time_to_sec(
            lines[-1].split(" : ")[1].split("(")[1].split(",")[1].split(")")[0]
        )
        # if end > duration:
        #         duration = end

    return beginning, end


def extract_cropped_images(video: str):
    video_name = video.split("\\")[-1].split(".")[0]

    if not os.path.exists(f"{IMAGES_PATH}/{video_name}/"):
        os.mkdir(f"{IMAGES_PATH}/{video_name}/")

    cap = cv2.VideoCapture(video)
    success, frame = cap.read()
    count = 0

    h0, h1, w0, w1 = crop_image(frame)
    # print(h0, h1, w0, w1)

    frame_rate = round(cap.get(cv2.CAP_PROP_FPS))
    width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(video_name, frame_rate, width, height)

    if video_name == "PW08182015LB":
        new_h0 = 30
        new_h1 = 1044
        new_w0 = 297
        new_w1 = 1610

    else:
        new_h0 = h0
        new_h1 = h1
        new_w0 = int(w0 + (w1 - w0) / 2 - 1.3 * (h1 - h0) / 2)
        new_w1 = int(w1 - (w1 - w0) / 2 + 1.3 * (h1 - h0) / 2)

    dim = (new_w1 - new_w0, new_h1 - new_h0)
    new_dim = (int(350 * dim[0] / dim[1]), 350)

    starts, ends = trim_video(video=video)

    while success:
        if count >= starts * frame_rate and count < (ends + 1) * frame_rate:
            if count % (frame_rate / 6) == 0:  # imags are saved at 6 fps

                new_frame = frame[new_h0:new_h1, new_w0:new_w1]
                frame_resized = cv2.resize(
                    new_frame, new_dim, interpolation=cv2.INTER_CUBIC
                )
                p = f"{IMAGES_PATH}/{video_name}/{video_name}Frame{str(count - starts*frame_rate).zfill(5)}.jpg"
                if not os.path.exists(p):
                    cv2.imwrite(
                        p,
                        frame_resized,
                    )
        success, frame = cap.read()
        count += 1


if __name__ == "__main__":
    videos_mp4 = glob.glob(f"{VIDEOS_PATH}/*/*.mp4", recursive=True)
    videos_mov = glob.glob(f"{VIDEOS_PATH}/*/*.mov", recursive=True)

    videos = videos_mov + videos_mp4
    print(len(videos))
    # print(videos)

    # videos = ['D:/Research/GJ DL/4 - New Annotations/GJ01092017DC.mp4']
    ann = []
    for d in glob.glob(f"{ANNOTATION_PATH}/*/"):
        ann.append(d.split("\\")[-2])
    print(len(ann))
    # print(ann)

    final_videos = []
    for v in videos:
        # if str(v).__contains__("PW06212016JK "):
        print(v)
        if v.split("\\")[-1] in ann:
            final_videos.append(v)
        else:
             print('err: ',v)   
    print(len(final_videos))
    # print(final_videos)

    # final_videos = ['D:/Research/GJ DL/5- New Videos\\2017\\GJ01092017DC.mp4']
    Parallel(n_jobs=60)(delayed(extract_cropped_images)(vid)
    # Parallel(n_jobs=1)(delayed(extract_cropped_images)(vid)
                        for vid in final_videos)
