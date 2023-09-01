import logging
import os

import uvicorn

from demo_annotation_tool.annotation_tool import AnnotatatorModel
from learning_loop_node.annotation.annotator_node import AnnotatorNode

logging.basicConfig(level=logging.DEBUG)

tool = AnnotatatorModel()
node = AnnotatorNode(name=f'Demo tool  {os.uname()[1]}', uuid='00000000-1111-2222-3333-555555555555', tool=tool)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', use_colors=True, reload=True)
