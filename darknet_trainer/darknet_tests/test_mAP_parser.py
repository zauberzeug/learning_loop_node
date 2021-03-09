from mAP_parser import MAPParser

data = """

(next mAP calculation at 1000 iterations) 
 1: 46.133102, 58.463966 avg loss, 3e-06 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left
 total_bbox = 142060, rewritten_bbox = 0.026045 % 
 total_bbox = 142063, rewritten_bbox = 0.026045 % 
 total_bbox = 142066, rewritten_bbox = 0.026044 % 
 total_bbox = 142068, rewritten_bbox = 0.026044 % 
 total_bbox = 142070, rewritten_bbox = 0.026043 % 
 total_bbox = 142073, rewritten_bbox = 0.026043 % 
 total_bbox = 142076, rewritten_bbox = 0.026042 % 
 total_bbox = 142078, rewritten_bbox = 0.026042 % 
 total_bbox = 142081, rewritten_bbox = 0.026041 % 
 total_bbox = 142084, rewritten_bbox = 0.026041 % 
 total_bbox = 142085, rewritten_bbox = 0.026041 % 
 total_bbox = 142088, rewritten_bbox = 0.026040 % 
 total_bbox = 142091, rewritten_bbox = 0.026040 % 
 total_bbox = 142092, rewritten_bbox = 0.026039 % 
 total_bbox = 142095, rewritten_bbox = 0.026039 % 
 total_bbox = 142098, rewritten_bbox = 0.026038 % 
 total_bbox = 142100, rewritten_bbox = 0.026038 % 
 total_bbox = 142103, rewritten_bbox = 0.026037 % 
 total_bbox = 142106, rewritten_bbox = 0.026037 % 
 total_bbox = 142108, rewritten_bbox = 0.026037 % 
 total_bbox = 142110, rewritten_bbox = 0.026036 % 
 total_bbox = 142112, rewritten_bbox = 0.026036 % 
 total_bbox = 142114, rewritten_bbox = 0.026035 % 
 total_bbox = 142116, rewritten_bbox = 0.026035 % 
 total_bbox = 142117, rewritten_bbox = 0.026035 % 
 total_bbox = 142120, rewritten_bbox = 0.026034 % 
 total_bbox = 142122, rewritten_bbox = 0.026034 % 
 total_bbox = 142125, rewritten_bbox = 0.026033 % 
 total_bbox = 142127, rewritten_bbox = 0.026033 % 
 total_bbox = 142130, rewritten_bbox = 0.026033 % 
 total_bbox = 142133, rewritten_bbox = 0.026032 % 
 total_bbox = 142134, rewritten_bbox = 0.026032 % 
 total_bbox = 142135, rewritten_bbox = 0.026032 % 
 total_bbox = 142138, rewritten_bbox = 0.026031 % 
 total_bbox = 142141, rewritten_bbox = 0.026030 % 
 total_bbox = 142143, rewritten_bbox = 0.026030 % 
 total_bbox = 142146, rewritten_bbox = 0.026030 % 
 total_bbox = 142148, rewritten_bbox = 0.026029 % 
 total_bbox = 142149, rewritten_bbox = 0.026029 % 
 total_bbox = 142152, rewritten_bbox = 0.026028 % 
 total_bbox = 142153, rewritten_bbox = 0.026028 % 
 total_bbox = 142156, rewritten_bbox = 0.026028 % 
 total_bbox = 142157, rewritten_bbox = 0.026028 % 
 total_bbox = 142158, rewritten_bbox = 0.026027 % 
 total_bbox = 142160, rewritten_bbox = 0.026027 % 
 total_bbox = 142162, rewritten_bbox = 0.026027 % 
 total_bbox = 142165, rewritten_bbox = 0.026026 % 
 total_bbox = 142167, rewritten_bbox = 0.026026 % 
 total_bbox = 142170, rewritten_bbox = 0.026025 % 
 total_bbox = 142172, rewritten_bbox = 0.026025 % 
 total_bbox = 142174, rewritten_bbox = 0.026024 % 
 total_bbox = 142177, rewritten_bbox = 0.026024 % 
 total_bbox = 142179, rewritten_bbox = 0.026024 % 
 total_bbox = 142182, rewritten_bbox = 0.026023 % 
 total_bbox = 142184, rewritten_bbox = 0.026023 % 
 total_bbox = 142187, rewritten_bbox = 0.026022 % 
 total_bbox = 142188, rewritten_bbox = 0.026022 % 
 total_bbox = 142191, rewritten_bbox = 0.026021 % 
 total_bbox = 142194, rewritten_bbox = 0.026021 % 
 total_bbox = 142197, rewritten_bbox = 0.026020 % 
 total_bbox = 142199, rewritten_bbox = 0.026020 % 
 total_bbox = 142202, rewritten_bbox = 0.026019 % 
 total_bbox = 142205, rewritten_bbox = 0.026019 % 
 total_bbox = 142207, rewritten_bbox = 0.026018 % 
Loaded: 0.000024 seconds

 (next mAP calculation at 1000 iterations) 
 2: 109.290443, 99.471283 avg loss, 0.000001 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left

1Total Detection Time: 0 Seconds
Saving weights to backup//tiny_yolo_best_mAP_0.000000_iteration_1000_avgloss_-nan_.weights
Saving weights to backup//tiny_yolo_1000_avgloss_-nan_.weights
Saving weights to backup//tiny_yolo_last.weights
 
Resizing to initial size: 32 x 32  try to allocate additional workspace_size = 0.15 MB 
 CUDA allocate done! 

calculation mAP (mean average precision)...
 Detection layer: 16 - type = 28 
 Detection layer: 23 - type = 28 

 detections_count = 0, unique_truth_count = 3  
class_id = 0, name = purple, ap = 37.00%          (TP = 34, FP = 7, FN = 8) 
class_id = 1, name = green, ap = 0.00%           (TP = 12, FP = 6, FN = 14) 

 for conf_thresh = 0.25, precision = -nan, recall = 0.00, F1-score = -nan 
 for conf_thresh = 0.25, TP = 0, FP = 0, FN = 3, average IoU = 0.00 % 

 IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
 mean average precision (mAP@0.50) = 0.793866, or 0.00 % 

Set -points flag:
 `-points 101` for MS COCO 
 `-points 11` for PascalVOC 2007 (uncomment `difficult` in voc.data) 
 `-points 0` (AUC) for ImageNet, PascalVOC 2010-2012, your custom dataset

 mean_average_precision (mAP@0.5) = 0.000000 

 """

parser = MAPParser(data)


def test_parse_mAP():
    mAP = parser.parse_mAP()
    expected_mAP = {'mAP': 0.793866, 'mAP_percentage': 0.5}

    assert mAP == expected_mAP


def test_parse_iteration():
    iteration = parser.parse_iteration()
    assert iteration == 2


def test_parse_classes():
    classes = parser.parse_classes()
    assert len(classes) == 2
    assert classes[0]['name'] == "green"
    assert classes[1]['name'] == "purple"


def test_parse_class():

    parsed_class = parser._parse_class('class_id = 0, name = purple, ap = 37.00%          (TP = 34, FP = 7, FN = 8)')

    assert parsed_class['id'] == "0"
    assert parsed_class['name'] == "purple"
    assert parsed_class['ap'] == 37
    assert parsed_class['tp'] == 34
    assert parsed_class['fp'] == 7
    assert parsed_class['fn'] == 8


def test_parse_training_status():
    assert parser.parse_training_status() == {
        'avg_loss': 99.471283, 'iteration': 2, 'loss': 109.290443, 'rate': 0.000001}
