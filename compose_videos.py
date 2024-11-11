import numpy as np
import matplotlib.pyplot as plt
import os
import apriltag
import skimage as ski
from skimage.morphology import erosion, square, dilation, binary_dilation, binary_erosion, disk
import cv2

def compose_videos(video_1, video_2, video_out):
    cap_1 = cv2.VideoCapture(video_1)
    cap_2 = cv2.VideoCapture(video_2)

    # Check if video opened successfully
    if not cap_1.isOpened():
        print("Error: Could not open video 1.")
        exit()
    if not cap_2.isOpened():
        print("Error: Could not open video 2.")
        exit()

    # Get the video's properties
    frame_width_1 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_1 = int(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width_2 = int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height_2 = int(cap_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_1.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(video_out, fourcc, fps, (frame_width_1 + frame_width_2, frame_height_1), isColor=True)

    # Process each frame
    while cap_1.isOpened():
        # Read the next frame from the video
        ret_1, frame_1 = cap_1.read()
        ret_2, frame_2 = cap_2.read()

        # Break the loop if no frame is captured (end of video)
        if not ret_1:
            break

        target = np.zeros(shape=(frame_height_1, frame_width_1 + frame_width_2, 3), dtype=frame_1.dtype)
        target[:, :frame_width_1, :] = frame_1
        target[:, frame_width_1: , :] = frame_2

        # Write the processed frame to the output video
        out.write(target)

    # Release resources
    cap_1.release()
    cap_2.release()
    out.release()

if __name__=="__main__":
    name='apriltags_p2'
    video_1=f'videos_detection_erosion/{name}_processed_erosion.mp4'
    video_2=f'videos_nofp_detection_erosion/{name}_nofp_detection_erosion.mp4'
    video_out=f'videos_detection_erosion_vs_npfp_detection_erosion/{name}_comparison.mp4'
    compose_videos(video_1, video_2, video_out)