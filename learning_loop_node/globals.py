import os


class Globals():
    def __init__(self) -> None:
        self.data_folder: str = os.getenv('DATA_FOLDER', '/data')
        self.detector_port: int = 5004  # NOTE used for tests


GLOBALS = Globals()
