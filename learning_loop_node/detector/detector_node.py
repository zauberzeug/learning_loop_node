from learning_loop_node.node import Node
from learning_loop_node.status import State


class DetectorNode(Node):

    async def send_status(self):
        # NOTE not yet implemented
        pass

    def get_state(self):
        return State.Running
