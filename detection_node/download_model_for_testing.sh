#!/bin/bash

echo "NOTE: call this script from the host."

TARGET_PATH='../detection_node_data'

rsync -ravh --copy-links prod:~/learning-loop-dev/projects/absammel_roboter/models/yolo4_tiny_3lspp_12_76844/ $TARGET_PATH
mv $TARGET_PATH/training_best_mAP_0.898879_iteration_76844_avgloss_2.452776_.weights $TARGET_PATH/some_weightfile.weights
rsync -ravh --copy-links prod:~/learning-loop-dev/projects/absammel_roboter/data/2021-01-17/j16/2462abd538f8_2021-01-17_08-33-49.800.jpg $TARGET_PATH