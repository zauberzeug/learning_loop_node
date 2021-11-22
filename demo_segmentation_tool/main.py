import uvicorn
import logging
from annotation_node import AnnotationNode
from demo_segmentation_tool import DemoSegmentationTool
import os

logging.basicConfig(level=logging.DEBUG)

tool = DemoSegmentationTool()
node = AnnotationNode(name=f'Demo tool  {os.uname()[1]}', uuid='00000000-1111-2222-3333-555555555555', tool=tool)

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
