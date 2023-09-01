from abc import abstractmethod
from typing import Optional

# pylint: disable=no-name-in-module
from pydantic import BaseModel

from learning_loop_node.data_classes import (AnnotationData,
                                             SegmentationAnnotation)


class UserInput(BaseModel):
    frontend_id: str
    data: AnnotationData


class ToolOutput(BaseModel):
    svg: str
    annotation: Optional[SegmentationAnnotation] = None


class AnnotatatorModel(BaseModel):

    @abstractmethod  # TODO: Auch bei anderen
    async def handle_user_input(self, user_input: UserInput, history: dict) -> ToolOutput:
        pass

    @abstractmethod
    def create_empty_history(self) -> dict:
        pass

    @abstractmethod
    def logout_user(self, sid: str):
        pass


class EmptyAnnotatator(AnnotatatorModel):
    async def handle_user_input(self, user_input: UserInput, history: dict) -> ToolOutput:
        return ToolOutput(svg="")

    def create_empty_history(self) -> dict:
        return {}

    def logout_user(self, sid: str) -> bool:
        return True
