# Detection scripts

## pupil-apriltags / apriltags3

### Related scripts
- *pupil_video_detect.py*

1. This script process videos accordingly to config. Config allows to to run the script in several modes. Config is placed at the top of the script and can be edited with your favourite text editor or IDE.
```
"""
MAIN CONFIG
"""
###########################
which_videos_to_process = 'all' # 'apriltag_1'
# filters
apply_threshold_filter = True
apply_shape_filter = True

save_video = True
save_path_suffix = "_detection_pupil_threshold_shape2"

display_detection_results = False
display_all_frames = False # if false it displays only frames with detections
save2json_rejected_ids = False
###########################
```
It is pretty straightforward. 
- It can be specified which videos are to be processed by their name (e.g. apriltags_new) or can be all videos from ./videos directiory by using *which_videos_to_process = 'all'*
- Using filters is simplified to booleans (e.g. if you wanna use threshold filter just set *apply_threshold_filter = True*)
- *save_video* is self descriptive
- *save_path_suffix* is used in order to specify output directory (e.g. *save_path_suffix = "_detection_pupil_threshold_shape2"* means that results will be saved at *./videos_detection_pupil_threshold_shape2*). At this location processed videos and results (in json format) will be saved. It is recommended to change path suffix when you change config of filters in order to keep everything well organized.
- *display_detection_results = True*  displays frames with detections, other frames will be skipped. To proceed to next detection result press any key having active displaying window.
- *display_all_frames = True* - displays all frames if *display_detection_results = True*
- *save2json_rejected_ids* - saves to the results file also ids that were filtered. Recommended for better insight into raw results, not recommended for further use of results.
2. Running the script:
```
uv run pupil_video_detect.py
```
By running this script videos are processed, logs mainly related to processing progress and filtering are printed to terminal and results of processing and detecting are saved.
3. Results can take form of .json or processed videos and .json depending upon script config. All results are saved to the ./*videos_<save_path_suffix>*. Videos display detection results including filtered detections and detections marked as valid, it also contains additional data like frame number, name of the video etc. Finally, .json files describes what ids were detected at given frame.
You can copy these results directiories containing .json files to the *./videos_analysis* for further processing of results.
## apriltags

### Related scripts
- sth.py
- 
```
sth.py
```

## ai approach

### Related scripts
- sth.py
- 
```
sth.py
```
# Results
### Related scripts
- *generate_all_tp_tn_fp_fn.sh*

Results are supposed to be stored in *./videos_analysis*. Structure is as follows:
- there are directories with names containing information about used script and its config (like in main, root directory). These directiores contain .json files which are raw detection results frame by frame.
- there is a directory containing ground truth of the same format
- there are some useful scripts to process results (you need to use only one which uses other scripts)

In order to process all results just execute following command
```bash
generate_all_tp_tn_fp_fn.sh
```
It produces all_results.json file which contains processed results in form of 
```
    {
        "apriltags_p1_detection": {
            "tp": 1504,
            "tn": 288,
            "fp": 4,
            "fn": 523
        }
    },
```
of all examined results. There is exemplary jupyter notebook file which can be used as a hint on how to load these results for further processing or visualization.