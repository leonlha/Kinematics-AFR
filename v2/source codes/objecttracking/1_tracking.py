# from ultralytics import YOLO
# import os
# import argparse

# # Load a model

# model = YOLO(r"best.pt")  # load an official segmentation model

# # root_directory = r'D:\Research\ToolTracking'

# parser = argparse.ArgumentParser(description="Process videos in a directory.")
# parser.add_argument("--r", type=str, required=True, help="Root directory containing video files")
# args = parser.parse_args()

# root_directory = args.r
# video_files = []  # List to store all video file paths

# def process_video(file_path):
#     # Process the video using the ByteTrack model
#     for results in model.track(source=file_path,
#                                 save_txt=True, save=True, show=False,
#                                 tracker="bytetrack.yaml", stream=True,conf=0.5):
#         # pass
#         # Check if there are no more results (end of video)
#         if results is None:
#             break  # Break out of the loop when there are no more results

# # Iterate through subfolders in the root directory
# for subfolder in os.listdir(root_directory):
#     subfolder_path = os.path.join(root_directory, subfolder)
    
#     # Check if the item is a directory
#     if os.path.isdir(subfolder_path):
#         # Iterate through video files inside the subfolder and append their paths to the list
#         video_files.extend([os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path) if filename.endswith(('.avi', '.mp4', '.mkv'))])

# print(len(video_files))

# # Parallel(n_jobs=-1)(delayed(process_video)(file_path) for file_path in video_files)
# for file_path in video_files:
#     process_video(file_path)

import os
import argparse
import torch
from ultralytics import YOLO
import cv2
from tqdm import tqdm


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process videos in a directory.")
parser.add_argument("--r", type=str, required=True, help="Root directory containing video files")
parser.add_argument("--g", type=str, required=False, default="0", help="GPU to use (e.g., '0', '0,1', etc.)")
args = parser.parse_args()

# # Set CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

# Function to process a video
def process_video(file_path):

    # Load a model
    model = YOLO(r"best.pt")  # load an official segmentation model 

    frame_counter = 0

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(file_path)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Release the video capture object
    cap.release()

    print('file:', file_path, 'total frame:', total_frames)

    # # Calculate the estimated total processing time
    # estimated_processing_time = total_frames * 0.02
    # print(f"Estimated total processing time: {estimated_processing_time:.2f} seconds assumming a speed of 20ms/frame")

    # Process the video using the YOLO model
    with tqdm(total=total_frames, desc='Processing Frames') as pbar:
        for results in model.track(source=file_path,
                                    save_txt=True,
                                    save=False,
                                    show=False,
                                    tracker="bytetrack.yaml",
                                    stream=True,
                                    verbose=False,
                                    conf=0.5):
            
            # Check if tracking is finished
            if results is None:
                break  # Break out of the loop when there are no more results

            # Check if the frame counter has exceeded the maximum frames
            if frame_counter > total_frames + 10:
                print('Error for file:', file_path)
                break  # Break out of the loop if maximum frames limit is reached

            # Increment the frame counter for each set of results
            frame_counter += 1
            # Update tqdm progress bar
            pbar.update(1)

        print('number of frames were tracked:', frame_counter)
    
# Root directory containing video files
root_directory = args.r

# List to store all video file paths
video_files = []

# Iterate through subfolders in the root directory
for subfolder in os.listdir(root_directory):
    subfolder_path = os.path.join(root_directory, subfolder)
    print(subfolder_path)

    # Check if the item is a directory
    if os.path.isdir(subfolder_path):
        # Iterate through video files inside the subfolder and append their paths to the list
        video_files.extend([os.path.join(subfolder_path, filename) for filename in os.listdir(subfolder_path) if filename.endswith(('.avi', '.mp4', '.mkv'))])

print((video_files))

# Process each video file
for file_path in video_files:
    process_video(file_path)

