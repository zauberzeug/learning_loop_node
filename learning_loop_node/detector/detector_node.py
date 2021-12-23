from learning_loop_node.detector import operation_mode
from learning_loop_node.detector.operation_mode import OperationMode
from learning_loop_node.node import Node
from learning_loop_node.status import State
from learning_loop_node.status import DetectionStatus, State
from learning_loop_node.context import Context
from learning_loop_node.rest import downloads
from fastapi.encoders import jsonable_encoder
from fastapi_utils.tasks import repeat_every
from typing import List, Optional
import shutil
import os
import logging
from learning_loop_node.globals import GLOBALS
from icecream import ic
import subprocess
from learning_loop_node.detector.detector import Detector
import asyncio
from learning_loop_node.detector.rest import detect
from learning_loop_node.detector.rest import upload
import numpy as np
from fastapi_socketio import SocketManager
from . import Detections
from . import Outbox
from threading import Thread


class DetectorNode(Node):
    detector: Detector
    organization: str
    project: str
    operation_mode: OperationMode = OperationMode.Idle
    outbox: Outbox

    target_model_id: Optional[str]

    def __init__(self, name: str, detector: Detector, uuid: str = None):
        super().__init__(name, uuid)
        self.detector = detector
        self.organization = os.environ.get('LOOP_ORGANIZATION', None) or os.environ.get('ORGANIZATION', None)
        self.project = os.environ.get('LOOP_PROJECT', None) or os.environ.get('PROJECT', None)
        assert self.organization, 'Detector node needs an organization'
        assert self.project, 'Detector node needs an project'
        self.outbox = Outbox()
        self.include_router(detect.router, tags=["detect"])
        self.include_router(upload.router, prefix="")
        self.include_router(operation_mode.router, tags=["operation_mode"])

        @self.on_event("startup")
        @repeat_every(seconds=1, raise_exceptions=False, wait_first=False)
        async def _check_for_update() -> None:
            await self.check_for_update()

        @self.on_event("startup")
        async def _load_model() -> None:
            try:
                self.detector.load_model()
            except:
                pass
            await self.check_for_update()

        @self.on_event("startup")
        @repeat_every(seconds=30, raise_exceptions=False, wait_first=False)
        def submit() -> None:
            thread = Thread(target=self.outbox.upload)
            thread.start()

        sio = SocketManager(app=self)

        @self.sio.on("detect")
        async def _detect(sid, data) -> None:
            try:
                np_image = np.frombuffer(data['image'], np.uint8)
                return await self.get_detections(np_image, data.get('mac', None), data.get('tags', []), data.get('active_learning', True))
            except Exception as e:
                logging.exception('could not detect via socketio')
                with open('/tmp/bad_img_from_socket_io.jpg', 'wb') as f:
                    f.write(data['image'])
                return {'error': str(e)}

        @self.sio.on("info")
        async def _info(sid) -> None:
            if self.detector.current_model:
                return self.detector.current_model.__dict__
            return 'No model loaded'

    async def check_for_update(self):
        try:
            logging.info(f'periodically checking operation mode. Currently the mode is {self.operation_mode}')
            if self.detector.current_model:
                logging.info(
                    f'Current model : { self.detector.current_model.version} with id { self.detector.current_model.id}')
            else:
                logging.info(f'Current model is None')
            if self.operation_mode == OperationMode.Check_for_updates:
                logging.info('going to check for new updates')
                await self.send_status()
                logging.info('Check if the current model id matches the target_model_id')

                if not self.detector.current_model or self.target_model_id != self.detector.current_model.id:
                    logging.info('No match. Going to download new version')
                    model_symlink = f'{GLOBALS.data_folder}/model'
                    target_model_folder = f'{GLOBALS.data_folder}/models/{self.target_model_id}'
                    shutil.rmtree(target_model_folder, ignore_errors=True)
                    os.makedirs(target_model_folder)
                    await downloads.download_model(target_model_folder, Context(organization=self.organization, project=self.project), self.target_model_id, self.detector.model_format)
                    try:
                        os.unlink(model_symlink)
                        os.remove(model_symlink)
                    except:
                        pass
                    os.symlink(target_model_folder, model_symlink)
                    logging.info(f'Updated symlink for model to {os.readlink(model_symlink)}')
                    self.reload()
                else:
                    logging.info('Versions are identic. Nothing to do.')
            else:
                logging.info('###### Operation mode was NOT check_for_updates')
        except Exception as e:
            logging.error(f'An error occured during "check_for_update". {str(e)}')
            raise

    async def send_status(self) -> dict:
        current_model_id = None
        try:
            current_model_id = self.detector.current_model.id
        except:
            pass
        status = DetectionStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            operation_mode=self.operation_mode,
            current_model_id=current_model_id,
            latest_error=self.status.latest_error,
        )
        logging.info(f'sending status {status}')
        response = await self.sio_client.call('update_detector', (self.organization, self.project, jsonable_encoder(status)), timeout=1)
        try:
            self.target_model_id = response['payload']['target_model_id']
            logging.info(f'After sending status. Target_model_id is {self.target_model_id}')
        except:
            logging.error('Could not send status to loop')
            return False
        return True

    def get_state(self):
        return State.Online

    async def set_operation_mode(self, mode: OperationMode):
        self.operation_mode = mode
        await self.send_status()

    def reload(self):
        subprocess.call(["touch", "/app/restart/restart.py"])

    async def get_detections(self, raw_image, mac: str, tags: str, active_learning=True):
        loop = asyncio.get_event_loop()
        # image = await loop.run_in_executor(None, lambda: cv2.imdecode(np_image, cv2.IMREAD_COLOR))
        detections = await loop.run_in_executor(None, self.detector.evaluate, raw_image)
        info = "\n    ".join([str(d) for d in detections.box_detections])
        logging.info(f'detected:\n    {info}')
        # if active_learning:
        #     thread = Thread(target=learn, args=(detections, mac, tags, image))
        #     thread.start()
        return jsonable_encoder(detections)

    async def upload_images(self, images: List[bytes]):
        loop = asyncio.get_event_loop()
        for image in images:
            await loop.run_in_executor(None, lambda: self.outbox.save(image, Detections(), ['picked_by_system']))
