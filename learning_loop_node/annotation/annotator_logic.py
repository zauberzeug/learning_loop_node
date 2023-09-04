from abc import abstractmethod

from learning_loop_node.data_classes import ToolOutput, UserInput


class AnnotatorLogic():

    @abstractmethod
    async def handle_user_input(self, user_input: UserInput, history: dict) -> ToolOutput:
        pass

    @abstractmethod
    def create_empty_history(self) -> dict:
        pass

    @abstractmethod
    def logout_user(self, sid: str):
        pass
