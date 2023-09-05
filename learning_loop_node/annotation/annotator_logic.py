from abc import abstractmethod
from typing import Dict, Optional

from ..data_classes import ToolOutput, UserInput
from ..node import Node


class AnnotatorLogic():

    def __init__(self):
        self._node: Optional[Node] = None

    def init(self, node: Node):
        self._node = node

    @abstractmethod
    async def handle_user_input(self, user_input: UserInput, history: Dict) -> ToolOutput:
        pass

    @abstractmethod
    def create_empty_history(self) -> Dict:
        pass

    @abstractmethod
    def logout_user(self, sid: str):
        pass
