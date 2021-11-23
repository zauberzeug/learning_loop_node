import os
from mock_annotation_tool import MockAnnotationTool
import uvicorn
import logging
from learning_loop_node.annotation_node.annotation_node import AnnotationNode
import backdoor_controls
logging.basicConfig(level=logging.DEBUG)
tool = MockAnnotationTool()
node = AnnotationNode(uuid='00000000-1111-2222-3333-444444444444', name=f'Annotation Node {os.uname()[1]}', tool=tool)

node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
