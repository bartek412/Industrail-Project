import numpy as np
import matplotlib.pyplot as plt
import os
import apriltag
import skimage as ski
from skimage.morphology import erosion, square, dilation, binary_dilation, binary_erosion, disk
import cv2

def count_detected_april_tags(image):
    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(options)
    results = detector.detect(np.asarray(image, np.uint8))
    return len(results)

def find_best_erosion_kernel(image, kernels):
    best_kernel = kernels[0]
    max_found_tags = 0
    best_kernel_iter = 0
    for iter in range(len(kernels)):
        mod_image = erosion(image, kernels[iter])
        found_tags = count_detected_april_tags(mod_image)
        if found_tags > max_found_tags:
            best_kernel = kernels[iter]
            max_found_tags = found_tags
            best_kernel_iter = iter
    print(best_kernel_iter)
    return best_kernel


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the video's properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(video_path.replace('.mp4', '_processed.mp4'), fourcc, fps, (frame_width, frame_height), isColor=True)

    # Process each frame
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is captured (end of video)
        if not ret:
            break

        original = frame.copy()

        # Process the frame (example: converting to grayscale)
        image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        kernel = find_best_erosion_kernel(image, [square(1), square(3), square(5), square(7), disk(3), disk(5), disk(7)])
        image = erosion(image, kernel)
        options = apriltag.DetectorOptions(families="tag16h5")
        detector = apriltag.Detector(options)
        results = detector.detect(np.asarray(image, np.uint8))

        # loop over the AprilTag detection results
        for r in results:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))
            # draw the bounding box of the AprilTag detection
            cv2.line(original, ptA, ptB, (0, 255, 0), 2)
            cv2.line(original, ptB, ptC, (0, 255, 0), 2)
            cv2.line(original, ptC, ptD, (0, 255, 0), 2)
            cv2.line(original, ptD, ptA, (0, 255, 0), 2)
            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            # cv2.circle(original, (cX, cY), 5, (0, 0, 255), -1)
            # draw the tag family on the image
            tagFamily = r.tag_family.decode("utf-8")
            cv2.putText(original, str(r.tag_id), (cX, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # Write the processed frame to the output video
        out.write(original)

    # Release resources
    cap.release()
    out.release()

if __name__=="__main__":
    video_path='samples/apriltags_p1.mp4'
    process_video(video_path)