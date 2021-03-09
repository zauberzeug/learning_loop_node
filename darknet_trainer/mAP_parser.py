class MAPParser:
    def __init__(self, log_file: str):
        self.log_file_lines = log_file.splitlines()

    def parse_mAP(self):
        # e.g. "mean average precision (mAP@0.50) = 0.793866, or 79.39 % ""
        for line in self.log_file_lines:
            line = line.strip()
            match = 'mean average precision'
            if line.startswith(match):
                mAP_percentage = line.split('mean average precision')[1].split('mAP@')[1].split(')')[0]
                mAP = line.split('mean average precision')[1].split(' = ')[1].split(', ')[0]
                return {"mAP": float(mAP), "mAP_percentage": float(mAP_percentage)}

        return

    def parse_iteration(self):
        for index, line in enumerate(self.log_file_lines):
            line = line.strip()
            match = '(next mAP calculation at'
            if line.startswith(match):
                iteration = int(self.log_file_lines[index + 1].split()[0].replace(':', ''))
                return iteration
        return

    def parse_classes(self):
        classes = []
        for line in self.log_file_lines:
            line = line.strip()
            if line.startswith("class_id"):
                a = line.split(", ")
                id = int(line.split("(")[1].split(", ")[2].split(" = ")[1].split(")")[0])
                classes.append(self._parse_class(line))
        return classes

    def parse_training_status(self):
        # e.g. 1061: 109.290443, 99.471283 avg loss, 0.000001 rate, 70.404934 seconds, 135808 images, 11070.078691 hours left
        for line in self.log_file_lines:
            line = line.strip()
            if "hours left" in line:
                data = line.split(", ")
                return {
                    "iteration": int(data[0].split(":")[0]),
                    "loss": float(data[0].split(":")[1]),
                    "avg_loss": float(data[1].split(" ")[0]),
                    "rate": float(data[2].split(" ")[0]),
                }

    def _parse_class(self, line):
        # e.g. class_id = 0, name = head, ap = 75.96%   	 (TP = 28, FP = 5, FN = 12)
        return {
            "id": line.split(", ")[0].split(" = ")[1],
            "name": line.split(", ")[1].split(" = ")[1],
            "ap": float(line.split(", ")[2].split(" = ")[1].split("%")[0]),
            "tp": int(line.split("(")[1].split(", ")[0].split(" = ")[1]),
            "fp": int(line.split("(")[1].split(", ")[1].split(" = ")[1]),
            "fn": int(line.split("(")[1].split(", ")[2].split(" = ")[1].split(")")[0])
        }
