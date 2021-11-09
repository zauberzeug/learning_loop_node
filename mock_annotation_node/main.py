import uvicorn
import logging
from annotation_node import AnnotationNode

logging.basicConfig(level=logging.DEBUG)


node = AnnotationNode()

if __name__ == "__main__":
    uvicorn.run("main:node", host="0.0.0.0", port=80, lifespan='on', reload=True)
