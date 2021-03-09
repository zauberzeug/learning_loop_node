from typing import List


class MAPParser:
    def __init__(self, iteration_log_lines: List[str]):
        self.iteration_log_lines = iteration_log_lines

    @staticmethod
    def extract_iteration_log(log: str) -> List[str]:
        match = '(next mAP calculation at'
        last_index = None
        pre_last_index = None
        log_lines = log.splitlines()
        for i in range(len(log_lines) - 1, 0, -1):
            line = log_lines[i]
            if match in line:
                if not last_index:
                    last_index = i
                elif not pre_last_index:
                    pre_last_index = i
                    break
        if not pre_last_index:
            return None
        return log_lines[pre_last_index:last_index]

    def parse_mAP(self):
        match = 'mean average precision'
        # e.g. "mean average precision (mAP@0.50) = 0.793866, or 79.39 % ""
        for line in self.iteration_log_lines:
            if line.strip().startswith(match):
                mAP_percentage = line.split('mean average precision')[1].split('mAP@')[1].split(')')[0]
                mAP = line.split('mean average precision')[1].split(' = ')[1].split(', ')[0]
                return {"mAP": float(mAP), "mAP_percentage": float(mAP_percentage)}

        return

    def parse_iteration(self):
        match = 'hours left'
        #  2: 109.290443, 99.471283 avg loss, 0.000001 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left
        for line in self.iteration_log_lines:
            if line.endswith(match):
                iteration = int(line.split()[0].replace(':', ''))
                return iteration
        return

    def parse_classes(self):
        classes = []
        for line in self.iteration_log_lines:
            if line.startswith("class_id"):
                classes.append(MAPParser._parse_class(line))
        return classes

    @staticmethod
    def _parse_class(line):
        # e.g. class_id = 0, name = head, ap = 75.96%   	 (TP = 28, FP = 5, FN = 12)
        return {
            "id": line.split(", ")[0].split(" = ")[1],
            "name": line.split(", ")[1].split(" = ")[1],
            "ap": float(line.split(", ")[2].split(" = ")[1].split("%")[0]),
            "tp": int(line.split("(")[1].split(", ")[0].split(" = ")[1]),
            "fp": int(line.split("(")[1].split(", ")[1].split(" = ")[1]),
            "fn": int(line.split("(")[1].split(", ")[2].split(" = ")[1].split(")")[0])
        }
