import uvicorn
import logging
from annotation_node import AnnotationNode
import backdoor_controls
logging.basicConfig(level=logging.DEBUG)


node = AnnotationNode()

node.include_router(backdoor_controls.router, prefix="")

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
