import asyncio
from threading import Thread
from learning_loop_node.trainer.capability import Capability
from learning_loop_node.trainer.trainer_node import TrainerNode
from darknet_trainer import DarknetTrainer
import uvicorn


darknet_trainer = DarknetTrainer(capability=Capability.Box)
trainer_node = TrainerNode(uuid='c34dc41f-9b76-4aa9-8b8d-9d27e33a19e4', name='darknet trainer', trainer=darknet_trainer)


@trainer_node.on_event("shutdown")
async def shutdown():

    def restart():
        asyncio.create_task(trainer_node.sio.disconnect())

    Thread(target=restart).start()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    uvicorn.run("main:trainer_node", host="0.0.0.0", port=80, lifespan='on', reload=True)
