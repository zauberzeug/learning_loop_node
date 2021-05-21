from typing import List
import helper
import os
from glob import glob
import re
import subprocess


def replace_classes_and_filters(classes_count: int, training_folder: str) -> None:
    cfg_file = _find_cfg_file(training_folder)

    with open(cfg_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('filters='):
            last_known_filters_line = i
        if line.startswith('[yolo]'):
            lines[last_known_filters_line] = f'filters={(classes_count+5)*3}'
            last_known_filters_line = None
        if line.startswith('classes='):
            lines[i] = f'classes={classes_count}'

    with open(cfg_file, 'w') as f:
        f.write('\n'.join(lines))


def update_anchors(training_folder: str) -> None:
    cfg_file_path = _find_cfg_file(training_folder)
    yolo_layer_count = _read_yolo_layer_count(cfg_file_path)
    width, height = _read_width_and_height(cfg_file_path)

    anchors = _calculate_anchors(training_folder, yolo_layer_count, width, height)
    _write_anchors(cfg_file_path, anchors)


def _find_cfg_file(folder: str) -> str:
    cfg_files = [file for file in glob(f'{folder}/**/*', recursive=True) if file.endswith('.cfg')]
    if len(cfg_files) == 0:
        raise Exception(f'[-] Error: No cfg file found.')
    elif len(cfg_files) > 1:
        raise Exception(f'[-] Error: Found more than one cfg file: {cfg_files}')
    return cfg_files[0]


def _calculate_anchors(training_path, yolo_layer_count: int, width: int, height: int):
    cmd = f'cd {training_path};/darknet/darknet detector calc_anchors data.txt -num_of_clusters {yolo_layer_count*3} -width {width} -height {height}'
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = p.communicate()
    if (p.returncode != 0):
        raise Exception(f'Calculating anchors failed:\nout: {out.decode("utf-8")} \nerror: {err.decode("utf-8") }')
    with open(f'{training_path}/anchors.txt', 'r') as f:
        anchors = f.readline()
    os.remove(f'{training_path}/anchors.txt')
    return anchors


def _read_yolo_layer_count(cfg_file_path: str):
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()

    yolo_layers = [line for line in lines
                   if line.lower().startswith('[yolo]')]
    return len(yolo_layers)


def _read_width_and_height(cfg_file_path: str):
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("width="):
            width = re.findall(r'\d+', line)[0]

        if line.startswith("height="):
            height = re.findall(r'\d+', line)[0]
    return width, height


def _write_anchors(cfg_file_path: str, anchors: str) -> None:
    with open(cfg_file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("anchors"):
            line = f'anchors={anchors}\n'
            lines[i] = line
    with open(cfg_file_path, 'w') as f:
        f.writelines(lines)
