import os
import cv2
import numpy as np
from pupil_apriltags import Detector
import json
import math
from skimage.draw import polygon

"""
MAIN CONFIG
"""
###########################
# which_videos_to_process = 'All'
which_videos_to_process = 'all' # 'apriltag_1', 'apriltags_new' 'apriltags_p1' 'apriltags_p2' 'apriltags_p3'

# filters
apply_threshold_filter = True
apply_shape_filter = True

apply_registry_filter = False # this filter was decided to be no longer used (but could be in the future, thus it was left here)

save_path_suffix = "_detection_pupil_threshold_shape2"
save_video = True
display_detection_results = False
display_all_frames = True # if false it displays only frames with detections
save2json_rejected_ids = False
###########################






# pupil apriltags detector config
TAG_FAMILY = "tag16h5"
NTHREADS = 1
QUAD_DECIMATE = 1.
QUAD_SIGMA = 0.8
REFINE_EDGES = 1
DECODE_SHARPENING = 0.45
DEBUG = 0

# filters variables - you can leave it as it is as it was empirically determined
DECISION_MARGIN_THRESHOLD = 2.4 # for threshold filter (value 0-100)
# you may ignore it - it is not a part of the project
DISCRIMINANT_INERTIA = 5 # used only if registry filter enabled (but this filter was decided to be no longer used (but could be in the future, thus it was left here))

print(f"""
        CONFIG:
        apply_threshold_filter : {apply_threshold_filter},
        Decision margin threshold: {DECISION_MARGIN_THRESHOLD},
        apply_shape_filter : {apply_shape_filter},
        save_video : {save_video},
        save2json_rejected_ids : {save2json_rejected_ids},
        Discriminant inertia: {DISCRIMINANT_INERTIA}
""")

at_detector = Detector(
    families=TAG_FAMILY,
    nthreads=NTHREADS,
    quad_decimate=QUAD_DECIMATE,
    quad_sigma=QUAD_SIGMA,
    refine_edges=REFINE_EDGES,
    decode_sharpening=DECODE_SHARPENING,
    debug=0
)

memorised_detections = np.empty([DISCRIMINANT_INERTIA, DISCRIMINANT_INERTIA])

def put_frame_number_and_video_title(original_frame, frame_number, num_of_frames_in_video, video_title):
    # add number of frame to image
    cv2.putText(img=original_frame, text=video_title, org=(10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 250, 10), thickness=1)
    cv2.putText(img=original_frame, text=str(frame_number) + '/' + str(num_of_frames_in_video), org=(10, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0, 255, 0), thickness=2)
    return original_frame

def display_detections(original_frame, detection_results, display_all_frames=False):
    if display_all_frames:
        cv2.imshow(f"AprilTags {len(detection_results)} - {video_src_path}", original_frame)
    elif len(detection_results) > 0:
        cv2.imshow(f"AprilTags {len(detection_results)} - {video_src_path}", original_frame)
    # wait for key press
    cv2.waitKey(0)

def put_detection_results_on_frame(original_frame, detection_results):
        for r in detection_results:
            tag_id = r.tag_id
            center = tuple(int(c) for c in r.center)
            corners = np.array(r.corners, dtype=np.int32)
            if isinstance(tag_id, int):
                cv2.polylines(original_frame, [corners], True, (0, 255, 0), 2)
                cv2.circle(original_frame, center, 5, (0, 0, 255), -1)
                cv2.putText(
                    original_frame,
                    f"ID: {tag_id}, \nD: {r.decision_margin:.2f}",
                    (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            else: # indicate ids classified as false positives
                cv2.polylines(original_frame, [corners], True, (255, 0, 0), 2)
                cv2.circle(original_frame, center, 5, (0, 155, 155), -1)
                cv2.putText(
                    original_frame,
                    f"ID: {tag_id}, \nD: {r.decision_margin:.2f}",
                    (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
        return original_frame

def threshold_filter(result):
    # loop over the AprilTag detection detection_results and reject unsure detection objects (below treshold)
    if result.decision_margin < DECISION_MARGIN_THRESHOLD:
        print(f"##### THRESHOLD FILTER ISSUE {result.tag_id}: result.decision_margin < DECISION_MARGIN_THRESHOLD = {result.decision_margin} < {DECISION_MARGIN_THRESHOLD}")
        result.tag_id = str(result.tag_id)
    return result

def count_idx(): # generator of indexes 0 to discriminant_inertia
    num = 0
    while True:
        yield num
        if num < DISCRIMINANT_INERTIA - 1:
            num += 1
        else:
            num = 0

def registry_filter(registry, last_frame_idx):
    num_of_ids = len(registry[last_frame_idx])
    flat_registry = []
    for r in registry:
        for rr in r:
            flat_registry.append(rr)
    flat_registry = np.array(flat_registry)
    ids_in_registry = np.unique(flat_registry)
    ids_occurences = np.empty_like(ids_in_registry)
    for i, id in enumerate(ids_in_registry):
        ids_occurences[i] = np.count_nonzero(flat_registry == id)
    most_frequent_ids_idxs = (-ids_occurences).argsort()[:num_of_ids]
    most_frequent_ids = ids_in_registry[most_frequent_ids_idxs]
    return most_frequent_ids

def calcTriangleArea(len_a, len_b, len_c):
    # using Heron's formula
    s = (len_a + len_b + len_c) / 2
    return math.sqrt(s * (s - len_a) * (s - len_b) * (s - len_c))

# side2side_ratios estimated empirically : 0.505 (for medium angle, 0,39 for low angle), side_square_to_area_min_ratio ratios estimated empirically : 0.505 (for medium angle (0.5, 2.6), (0.4, 2.75) for low angle)
def shape_filter(result, previous_results=None, side2side_ratio = 0.39, diagonal2diagonal_ratio = 0.4, side_square_to_area_min_ratio=(0.4, 2.75), opposite_sides_min_ratio=0.65):
    (ptA, ptB, ptC, ptD) = result.corners
    # sides lengths
    len_ab = math.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)
    len_bc = math.sqrt((ptB[0] - ptC[0]) ** 2 + (ptB[1] - ptC[1]) ** 2)
    len_cd = math.sqrt((ptC[0] - ptD[0]) ** 2 + (ptC[1] - ptD[1]) ** 2)
    len_da = math.sqrt((ptD[0] - ptA[0]) ** 2 + (ptD[1] - ptA[1]) ** 2)
    # diagonals lengths
    len_ac = math.sqrt((ptA[0] - ptC[0]) ** 2 + (ptA[1] - ptC[1]) ** 2)
    len_bd = math.sqrt((ptB[0] - ptD[0]) ** 2 + (ptB[1] - ptD[1]) ** 2)
    print(f"##### SHAPE FILTER DATA: len_ab = {len_ab}, len_bc = {len_bc},len_cd = {len_cd},len_da = {len_da}")
    print(f"ptA = {ptA}, ptB = {ptB}, ptC = {ptC}, ptD = {ptD}")

    # check all sides 
    # 1. compare lengths of sides 2. compare diagonals 3. compare areas

    # 1. We assume tag is somewhat a square, so its sides are expected to be more or less of equal length
    sides_lengths = [len_ab, len_bc, len_cd, len_da]
    if min(sides_lengths)/max(sides_lengths) < side2side_ratio:
        result.tag_id = str(result.tag_id)
        print(f"### 1.SHAPE FILTER ISSUE {result.tag_id}: min(sides_lengths)/max(sides_lengths) < side2side_ratio-> {min(sides_lengths)/max(sides_lengths)} < {side2side_ratio}")
        print(f"sides_lengths = {sides_lengths}")
        return result # True
        
    # 2. Diagonals comparison
    if min(len_ac/len_bd, len_bd/len_ac) < diagonal2diagonal_ratio:
        result.tag_id = str(result.tag_id)
        print(f"### 2.SHAPE FILTER ISSUE {result.tag_id}: min(len_ac/len_bd, len_bd/len_ac) < diagonal2diagonal_ratio: -> min({len_ac/len_bd, len_bd/len_ac}) < {diagonal2diagonal_ratio}")
        return result # True

    # 3. Detected tag area vs area if it were somewhat a square
    area = calcTriangleArea(len_ab, len_bd, len_da) + calcTriangleArea(len_bc, len_cd, len_bd)
    if np.min(np.array(sides_lengths)**2 / area) < side_square_to_area_min_ratio[0] or np.max(np.array(sides_lengths)**2 / area) > side_square_to_area_min_ratio[1]:
        result.tag_id = str(result.tag_id)
        print(f"### 3.SHAPE FILTER ISSUE {result.tag_id}: np.min(np.array(sides_lengths)**2 / area) < side_square_to_area_min_ratio or np.max(np.array(sides_lengths)**2 / area) > 1+(1-side_square_to_area_min_ratio) -> min({np.array(sides_lengths)**2 / area}) < {side_square_to_area_min_ratio[0]} or np.max{np.array(sides_lengths)**2 / area} > {side_square_to_area_min_ratio[1]}")
        return result # True         

    # 4. Opposite sides comparison
    if min(len_ab / len_cd, len_cd / len_ab) < opposite_sides_min_ratio:
        result.tag_id = str(result.tag_id)
        print(f"### 4.SHAPE FILTER ISSUE {result.tag_id}: min(len_ab / len_cd, len_cd / len_ab) < opposite_sides_min_ratio -> {min(len_ab / len_cd, len_cd / len_ab)} < {opposite_sides_min_ratio} = {min(len_ab / len_cd, len_cd / len_ab) < opposite_sides_min_ratio}")
        return result # True

    # 5. Mask
    if previous_results == None:
        return result # False
    return result # False

def process_video(video_src_path, save_video=False, apply_threshold_filter=True, apply_registry_filter=False, apply_shape_filter=False, display_detection_results=False):#, video_out_path, save=False):
    # Open video and check if video was opened successfully
    print(f"Opening: {video_src_path}")
    video = cv2.VideoCapture(video_src_path)
    if not video.isOpened():
        exit("Error: Could not open video.")

    video_saver = ':)'
    if save_video:
        # Get the video's properties
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
        video_saver = cv2.VideoWriter(
            video_out_path,
            fourcc,
            fps,
            (frame_width, frame_height),
            isColor=True,
        )

    previous_results = None
    stats = {"frames_numbers": 0, "frames_detected": 0}
    statistics = {}
    frames_counter = 0
    num_of_frames_in_video = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if apply_registry_filter:
        # initialize - fill ids_raw_registry
        ids_raw_registry = [':)' for d in range(DISCRIMINANT_INERTIA)] # raw detected ids (no swaps)
        idx_counter = count_idx()
        last_frame = len(ids_raw_registry) - 1  # leave one place for new frame
        for i in range(last_frame):
            reg_idx = next(idx_counter) # it counts from 0 to 4 to fill registry
            
            ret, frame = video.read()
            if not ret:
                exit(f"Video has to have at least {last_frame} frames")
            original_frame = frame.copy()
            stats["frames_numbers"] += 1
            statistics[frames_counter] = []
            image = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            detection_results = at_detector.detect(np.asarray(image, np.uint8))
            if apply_threshold_filter:
                detection_results = [threshold_filter(result=r) for r in detection_results]
            print(f'progress: {stats["frames_numbers"]} / {num_of_frames_in_video} ({stats["frames_numbers"] / num_of_frames_in_video * 100:.2f} %)')
            
            if apply_shape_filter:
                detection_results = [shape_filter(result=r, previous_results=previous_results) for r in detection_results]

            detected_id = [r.tag_id for r in detection_results] # raw id coming from detecor (may contain invalid detections) - so next frame detection result
            ids_raw_registry[i] = detected_id # load n frames to fill registry
            print(f"ids_raw_registry[{reg_idx}] << {detected_id}")

            for r in detection_results:
                 statistics[frames_counter].append(r.tag_id)

            if save_video:
                # Get frame with detection result superimposed on it and save
                original_frame = put_detection_results_on_frame(original_frame=original_frame, detection_results=detection_results, frame_number=frames_counter)
                video_saver.write(original_frame)
            previous_results = detection_results
            frames_counter += 1

    while video.isOpened():
        # Read the next frame from the video
        ret, frame = video.read()

        # Break the loop if no frame is captured (end of video)
        if not ret:
            break

        original_frame = frame.copy()

        stats["frames_numbers"] += 1
        statistics[frames_counter] = []

        print(f'progress: {stats["frames_numbers"]} / {num_of_frames_in_video} ({stats["frames_numbers"] / num_of_frames_in_video * 100:.2f} %)')
        # Process the frame (example: converting to grayscale)
        image = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        # detect
        detection_results = at_detector.detect(np.asarray(image, np.uint8))
        detected_id = []

        if apply_threshold_filter:
            detection_results = [threshold_filter(result=r) for r in detection_results]

        if apply_shape_filter:
            detection_results = [shape_filter(result=r, previous_results=previous_results) for r in detection_results]

        if apply_registry_filter and len(detection_results) == 1:
            detected_id = [r.tag_id for r in detection_results]
            reg_idx = next(idx_counter)
            ids_raw_registry[reg_idx] = detected_id
            print(f"ids_raw_registry[{reg_idx}] << {detected_id}")
            swap_id = registry_filter(registry=ids_raw_registry, last_frame_idx=reg_idx)[0]
            print(f"registryian_filter swap id {detection_results[0].tag_id} with {swap_id}")
            detection_results[0].tag_id = swap_id

        if save_video or display_detection_results:
            # Get frame with detection result superimposed on it and save
            original_frame = put_detection_results_on_frame(original_frame=original_frame, detection_results=detection_results)
            original_frame = put_frame_number_and_video_title(original_frame=original_frame, frame_number=stats["frames_numbers"], num_of_frames_in_video=num_of_frames_in_video, video_title=video_src_path)
            if save_video:
                video_saver.write(original_frame)
            if display_detection_results:
                display_detections(original_frame=original_frame, detection_results=detection_results, display_all_frames=display_all_frames)


        for r in detection_results:
            print(f"detected tag id {r.tag_id}, {type(r.tag_id)}")
            statistics[frames_counter].append(r.tag_id)
        

        previous_results = detection_results
        frames_counter += 1

    video.release()
    video_saver.release()
    print(10*'\n')
    if not save2json_rejected_ids:
        for frame, detections_within_frame in statistics.items():
            statistics[frame] = [detection for detection in detections_within_frame if isinstance(detection, int)]

    with open(video_out_path.replace('.mp4', '.json'), "w") as json_file:
                    json.dump(statistics, json_file, indent=2)

    return

if __name__ == "__main__":

    if which_videos_to_process == 'all':
        videos_dir = [v for v in os.listdir(os.getcwd()+'/videos') if v[-4:] == ".mp4"]
        print("videos found in directory:", os.getcwd()+'/videos', videos_dir)
        for name in videos_dir:
            video_src_path = f'videos/{name}'
            video_out_path = f'videos{save_path_suffix}/{name[:-4]}{save_path_suffix}.mp4'
            process_video(video_src_path, apply_threshold_filter=apply_threshold_filter, apply_registry_filter=apply_registry_filter, apply_shape_filter=apply_shape_filter, save_video=save_video, display_detection_results=display_detection_results)

    else:
        name = which_videos_to_process
        video_src_path = f'videos/{name}.mp4'
        video_out_path = f'videos{save_path_suffix}/{name}{save_path_suffix}.mp4'
        process_video(video_src_path, apply_threshold_filter=apply_threshold_filter, apply_registry_filter=apply_registry_filter, apply_shape_filter=apply_shape_filter, save_video=save_video, display_detection_results=display_detection_results)

