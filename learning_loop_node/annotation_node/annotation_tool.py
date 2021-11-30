from abc import abstractmethod
from pydantic import BaseModel
from learning_loop_node.annotation_node.data_classes import UserInput, ToolOutput


class AnnotationTool(BaseModel):

    @abstractmethod
    async def handle_user_input(self, user_input: UserInput, history: dict) -> ToolOutput:
        pass

    @abstractmethod
    def create_empty_history(self) -> dict:
        pass


class EmptyAnnotationTool():
    async def handle_user_input(self, user_input: UserInput, history: dict) -> ToolOutput:
        return ToolOutput(svg="", shape={})

    def create_empty_history(self) -> dict:
        return {}
