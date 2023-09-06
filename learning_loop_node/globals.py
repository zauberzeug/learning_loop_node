
class Globals():
    def __init__(self):
        self.data_folder: str = '/data'  # p? in env schreiben?
        self.detector_port: int = 5002
        self.mock_trainer_node_port = 5001


GLOBALS = Globals()
