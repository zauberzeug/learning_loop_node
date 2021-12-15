from learning_loop_node.node import Node
from learning_loop_node.status import State
import os

class DetectorNode(Node):

    def __init__(self, name: str, uuid: str = None):
        super().__init__(name, uuid)
        
        self.organization = os.environ.get('LOOP_ORGANIZATION', None) or os.environ.get('ORGANIZATION', None)
        self.project = os.environ.get('LOOP_PROJECT', None) or os.environ.get('PROJECT', None)
        assert self.organization, 'Detector node needs an organization'
        assert self.project, 'Detector node needs an project'

    async def send_status(self):
        # NOTE not yet implemented
        pass

    def get_state(self):
        return State.Running
