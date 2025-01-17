import json
import sys


if len(sys.argv) == 3:
    source_file_path = sys.argv[1]
    ground_truth_path = sys.argv[2]

confusion_matrix_splits = source_file_path.split('/')
name = confusion_matrix_splits[-1].replace(".json", "")
confusion_matrix_splits[-1] = f'confusion_matrix_{confusion_matrix_splits[-1]}'
confusion_matrix_path = confusion_matrix_splits[0]
if len(confusion_matrix_splits) > 0:
    for split in confusion_matrix_splits[1:]:
        confusion_matrix_path += f'/{split}'

class ConfusionMatrixSimplified:
    tp: 0
    tn: 0
    fp: 0
    fn: 0

    def __init__(self, tp = 0, tn = 0, fp = 0, fn = 0):
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def calculate(self, detections, ground_truth):
        for frame_no, _ in detections.items():
            detected_tags = detections[frame_no]
            ground_truth_tags = ground_truth[frame_no]

            if len(detected_tags) == 0 and len(ground_truth_tags) == 0:
                self.tn += 1
            for tag in detected_tags:
                if tag in ground_truth_tags:
                    self.tp += 1
                else:
                    self.fp += 1
            for tag in ground_truth_tags:
                if tag not in detected_tags:
                    self.fn += 1


compared_file = open(source_file_path)
ground_truth_file = open(ground_truth_path)

compared_detections = json.load(compared_file)
ground_truth = json.load(ground_truth_file)

compared_file.close()
ground_truth_file.close()

confusion_mat = ConfusionMatrixSimplified()
confusion_mat.calculate(compared_detections, ground_truth)

confusion_mat_file = open(confusion_matrix_path, 'w')
json.dump({name: confusion_mat.__dict__}, confusion_mat_file, indent=2)
confusion_mat_file.close()