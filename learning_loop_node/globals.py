from . import LoopCommunication


class Globals():
    def __init__(self):
        self.data_folder: str = '/data'
        self.loop_communication: LoopCommunication = LoopCommunication()


GLOBALS = Globals()
