from abc import abstractmethod
from typing import Dict, Optional

from ..data_classes import ToolOutput, UserInput
from ..node import Node


class AnnotatorLogic():

    def __init__(self) -> None:
        self._node: Optional[Node] = None

    def init(self, node: Node) -> None:
        self._node = node

    @abstractmethod
    async def handle_user_input(self, user_input: UserInput, history: Dict) -> ToolOutput:
        """ This method is called when a user input is received from the client.
            The function should return a ToolOutput object."""

    @abstractmethod
    def create_empty_history(self) -> Dict:
        """ This method is called when a new annotation session is started.
            The function should return an empty history object."""

    @abstractmethod
    def logout_user(self, sid: str) -> bool:
        """ This method is called when a user disconnects from the server.
            The function should return True if the user was logged out successfully."""
