import numpy as np
import apriltag
from skimage.morphology import (
    erosion,
    square,
    disk,
)
import cv2
import math
from skimage.draw import polygon
import json


def count_detected_april_tags(image):
    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(options)
    results = detector.detect(np.asarray(image, np.uint8))
    return len(results)


def find_best_erosion_kernel(image, kernels):
    best_kernel = kernels[0]
    max_found_tags = 0
    # best_kernel_iter = 0
    for iter in range(len(kernels)):
        mod_image = erosion(image, kernels[iter])
        found_tags = count_detected_april_tags(mod_image)
        if found_tags > max_found_tags:
            best_kernel = kernels[iter]
            max_found_tags = found_tags
    # print(best_kernel_iter)
    return best_kernel

def calcTriangleArea(len_a, len_b, len_c):
    # using Heron's formula
    s = (len_a + len_b + len_c) / 2
    return math.sqrt(s * (s - len_a) * (s - len_b) * (s - len_c))

def create_results_mask(shape, results):
    mask = np.zeros(shape, dtype=np.uint8)
    for result in results:
        xs = [corner[0] for corner in result.corners]
        ys = [corner[1] for corner in result.corners]
        yy_mask, xx_mask = polygon(ys, xs, shape)
        mask[yy_mask, xx_mask] = 1
    return mask

def is_false_positive_detection(result, previous_results=None, shape=None, side_square_to_area_min_ratio=0.95, opposite_sides_min_ratio=0.75):
    (ptA, ptB, ptC, ptD) = result.corners
    # sides lengths
    len_ab = math.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)
    len_bc = math.sqrt((ptB[0] - ptC[0]) ** 2 + (ptB[1] - ptC[1]) ** 2)
    len_cd = math.sqrt((ptC[0] - ptD[0]) ** 2 + (ptC[1] - ptD[1]) ** 2)
    len_da = math.sqrt((ptD[0] - ptA[0]) ** 2 + (ptD[1] - ptA[1]) ** 2)
    # diagonals lengths
    len_ac = math.sqrt((ptA[0] - ptC[0]) ** 2 + (ptA[1] - ptC[1]) ** 2)
    len_bd = math.sqrt((ptB[0] - ptD[0]) ** 2 + (ptB[1] - ptD[1]) ** 2)

    area = calcTriangleArea(len_ab, len_bd, len_da) + calcTriangleArea(len_bc, len_cd, len_bd)
    side_square_to_area_ratio = (((len_ab + len_bc + len_cd + len_da) / 4) ** 2) / area

    if side_square_to_area_ratio < side_square_to_area_min_ratio:
        return True

    if min(len_ab / len_cd, len_cd / len_ab) < opposite_sides_min_ratio:
        return True

    if min(len_bc / len_da, len_da / len_bc) < opposite_sides_min_ratio:
        return True

    if previous_results == None:
        return False

    result_mask = create_results_mask(shape, [result])
    previous_results_mask = create_results_mask(shape, previous_results)
    if np.sum(np.minimum(result_mask, previous_results_mask)) == 0:
        return True

    return False

def is_dark_frame(frame, px_value=30, ratio=0.3):
    return (np.sum((frame > px_value) * 1.0) / (frame.flatten().shape[0])) > ratio


def process_video(video_src_path, video_out_path, use_erosion=True, save=True, for_dark_only=True):
    cap = cv2.VideoCapture(video_src_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Get the video's properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(
        video_out_path,
        fourcc,
        fps,
        (frame_width, frame_height),
        isColor=True,
    )

    fp_detected = 0

    # Process each frame

    frames_counter = 0
    stats = {"frames_numbers": 0, "frames_detected": 0}
    statistics = {}
    previous_results = None
    while cap.isOpened():
        # Read the next frame from the video
        ret, frame = cap.read()

        # Break the loop if no frame is captured (end of video)
        if not ret:
            break

        original = frame.copy()


        statistics[frames_counter] = []
        stats["frames_numbers"] += 1
        print(f'progress: {stats["frames_numbers"]} / {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} ({stats["frames_numbers"] / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * 100:.2f} %)')
        # Process the frame (example: converting to grayscale)
        image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if use_erosion and (not is_dark_frame(image) or not for_dark_only):
            print('use erosion')
            kernel = find_best_erosion_kernel(
                image,
                [square(1), square(3), square(5), square(7), disk(3), disk(5), disk(7)],
            )
            image = erosion(image, kernel)
        options = apriltag.DetectorOptions(families="tag16h5")
        detector = apriltag.Detector(options)
        results = detector.detect(np.asarray(image, np.uint8))
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if results:
            stats["frames_detected"] += 1
        # loop over the AprilTag detection results
        for r in results:
            if is_false_positive_detection(r, previous_results, image.shape):
                fp_detected += 1
                print('continue')
                continue

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

            statistics[frames_counter].append(r.tag_id)

        # add number of frame to image
        cv2.putText(original, str(frames_counter), (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the processed frame to the output video
        if save:
            out.write(original)

        frames_counter += 1

        previous_results = results

    print(f'detected false positives: {fp_detected}')

    # Release resources
    cap.release()
    out.release()

    with open(video_out_path.replace('.mp4', '.json'), "w") as json_file:
        json.dump(statistics, json_file, indent=2)

    return stats


if __name__ == "__main__":
    name = 'apriltag_1'
    video_src_path = f'videos/{name}.mp4'
    video_out_path = f'videos_nofp_detection_erosion/{name}_nofp_detection_erosion.mp4'
    process_video(video_src_path, video_out_path, use_erosion=True, save=True, for_dark_only=True)
