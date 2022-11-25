from enum import auto
import multiprocessing
from . import Detections
from . import Outbox
from .rest.operation_mode import OperationMode
from .detector import Detector
from .rest import detect, upload, operation_mode
from ..socket_response import SocketResponse
from ..inbox_filter.relevance_filter import RelevanceFilter
from ..rest import downloads
from ..node import Node
from ..status import State
from ..status import DetectionStatus, State
from ..context import Context
from ..globals import GLOBALS
from ..data_classes.category import Category
from ..model_information import ModelInformation
from fastapi.encoders import jsonable_encoder
from fastapi_utils.tasks import repeat_every
from typing import List, Union
import os
import contextlib
import logging
import subprocess
import asyncio
import numpy as np
from fastapi_socketio import SocketManager
from threading import Thread
from datetime import datetime
from icecream import ic
import shutil
from .. import environment_reader
from multiprocessing import Event


class DetectorNode(Node):
    update_frequency = 10

    def __init__(self, name: str, detector: Detector, uuid: str = None):
        super().__init__(name, uuid)
        self.detector = detector
        self.organization = environment_reader.organization()
        self.project = environment_reader.project()
        assert self.organization, 'Detector node needs an organization'
        assert self.project, 'Detector node needs an project'
        self.log.info(f'Using {self.organization}/{self.project}')
        self.operation_mode: OperationMode = OperationMode.Startup
        self.connected_clients: List[str] = []

        self.outbox: Outbox = Outbox()

        self.relevance_filter: RelevanceFilter = RelevanceFilter(self.outbox)
        self.target_model = None
        self.include_router(detect.router, tags=["detect"])
        self.include_router(upload.router, prefix="")
        self.include_router(operation_mode.router, tags=["operation_mode"])

        @self.on_event("startup")
        @repeat_every(seconds=self.update_frequency, raise_exceptions=False, wait_first=False)
        async def _check_for_update() -> None:
            try:
                await self.check_for_update()
            except:
                self.log.exception("error in '_check_for_update'")

        @self.on_event("startup")
        async def startup() -> None:
            try:
                self.log.info("received 'startup' event")
                self.outbox.start_continuous_upload()
                self._load_model()
            except:
                self.log.exception("error during 'startup'")

        sio = SocketManager(app=self)

        @self.sio.on("detect")
        async def _detect(sid, data) -> None:
            try:
                np_image = np.frombuffer(data['image'], np.uint8)
                return await self.get_detections(
                    raw_image=np_image,
                    camera_id=data.get('camera-id', None) or data.get('mac', None),
                    tags=data.get('tags', []),
                    autoupload=data.get('autoupload', None),
                )
            except Exception as e:
                self.log.exception('could not detect via socketio')
                with open('/tmp/bad_img_from_socket_io.jpg', 'wb') as f:
                    f.write(data['image'])
                return {'error': str(e)}

        @self.sio.on("info")
        async def _info(sid) -> None:
            if self.detector.current_model:
                return self.detector.current_model.__dict__
            return 'No model loaded'

        @self.sio.on('upload')
        async def _upload(sid, data):
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, self.outbox.save, data['image'], Detections(), ['picked_by_system'])
            except Exception as e:
                self.log.exception('could not upload via socketio')
                return {'error': str(e)}

        @self.sio.event
        def connect(sid, environ, auth):
            self.connected_clients.append(sid)

        @self.on_event("shutdown")
        async def shutdown():
            await self.shutdown()

    async def shutdown(self):
        try:
            self.log.info("received 'shutdown' event")
            await self._disconnect_sio_clients()
            self.outbox.stop_continuous_upload()
        except:
            self.log.exception("error during 'shutdown'")

    def _load_model(self) -> None:
        try:
            self.detector.load_model()
        finally:
            self.operation_mode = OperationMode.Idle

    async def _disconnect_sio_clients(self):
        for sid in self.connected_clients:
            await self.sio.disconnect(sid)

    async def check_for_update(self):
        if self.operation_mode == OperationMode.Startup:
            return
        try:
            self.log.info(f'periodically checking operation mode. Current mode is {self.operation_mode}')
            update_to_model_id = await self.send_status()
            if self.detector.current_model:
                self.log.info(
                    f'Current model: {self.detector.current_model.version} with id {self.detector.current_model.id}')
            else:
                self.log.info(f'no model loaded')
            if self.operation_mode != OperationMode.Idle:
                self.log.info(f'not checking for updates; operation mode is {self.operation_mode}')
                return

            self.status.reset_error('update_model')
            if self.target_model is None:
                self.log.info(f'not checking for updates; no target model selected')
                return
            self.log.info('going to check for new updates')
            if not self.detector.current_model or self.target_model != self.detector.current_model.version:
                self.log.info(
                    f'Current model "{self.detector.current_model.version if self.detector.current_model else "-"}" needs to be updated to {self.target_model}')
                with pushd(GLOBALS.data_folder):
                    model_symlink = 'model'
                    target_model_folder = f'models/{self.target_model}'
                    shutil.rmtree(target_model_folder, ignore_errors=True)
                    os.makedirs(target_model_folder)

                    await downloads.download_model(
                        target_model_folder,
                        Context(organization=self.organization, project=self.project),
                        update_to_model_id,
                        self.detector.model_format
                    )
                    try:
                        os.unlink(model_symlink)
                        os.remove(model_symlink)
                    except:
                        pass
                    os.symlink(target_model_folder, model_symlink)
                    self.log.info(f'Updated symlink for model to {os.readlink(model_symlink)}')

                    await self.send_status()
                    self.reload(because='new model installed')
            else:
                self.log.info('Versions are identic. Nothing to do.')
        except Exception as e:
            self.log.exception(f'check_for_update failed')
            msg = e.cause if isinstance(e, downloads.DownloadError) else str(e)
            self.status.set_error('update_model', f'Could not update model: {msg}')
            await self.send_status()

    async def send_status(self) -> Union[str, bool]:
        if not self.sio_client.connected:
            self.log.error('could not send status -- we are not connected to the Learning Loop')
            return False
        status = DetectionStatus(
            id=self.uuid,
            name=self.name,
            state=self.status.state,
            errors=self.status._errors,
            uptime=int((datetime.now() - self.startup_time).total_seconds()),
            operation_mode=self.operation_mode,
            current_model=self.detector.current_model.version if self.detector.current_model else None,
            target_model=self.target_model,
            model_format=self.detector.model_format,
        )

        self.log.debug(f'sending status {status}')
        response = await self.sio_client.call('update_detector', (self.organization, self.project, jsonable_encoder(status)))
        socket_response = SocketResponse.from_dict(response)
        if not socket_response.success:
            self.log.error(f'Statusupdate failed: {response}')
            return False
        self.target_model = socket_response.payload['target_model_version']
        self.log.debug(f'After sending status. Target_model is {self.target_model}')
        return socket_response.payload['target_model_id']

    def get_state(self):
        return State.Online

    async def set_operation_mode(self, mode: OperationMode):
        self.operation_mode = mode
        await self.send_status()

    def reload(self, because: str):
        '''provide a cause for the reload'''

        print('########## reloading app because ' + because, flush=True)
        if os.path.isfile('/app/restart/restart.py'):
            subprocess.call(['touch', '/app/restart/restart.py'])
        else:
            subprocess.call(['touch', '/app/main.py'])

    async def get_detections(self, raw_image, camera_id: str, tags: str, autoupload: str = None):
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(None, self.detector.evaluate, raw_image)
        detections = self.add_category_id_to_detections(self.detector.model_info, detections)
        for detection in detections.segmentation_detections:
            detection.shape = ','.join([str(value) for p in detection.shape.points for _, value in p.__dict__.items()])
        info = "\n    ".join([str(d) for d in detections.box_detections +
                             detections.point_detections + detections.segmentation_detections])
        self.log.info(f'detected:\n    {info}')
        if camera_id is not None:
            tags.append(camera_id)
        if autoupload is None or autoupload == 'filtered':  # NOTE default is filtered
            Thread(target=self.relevance_filter.learn, args=(detections, camera_id, tags, raw_image)).start()
        elif autoupload == 'all':
            Thread(target=self.outbox.save, args=(raw_image, detections, tags)).start()
        elif autoupload == 'disabled':
            pass
        else:
            self.log.warning(f'unknown autoupload value {autoupload}')
        return jsonable_encoder(detections)

    async def upload_images(self, images: List[bytes]):
        loop = asyncio.get_event_loop()
        for image in images:
            await loop.run_in_executor(None, self.outbox.save, image, Detections(), ['picked_by_system'])

    def add_category_id_to_detections(self, model_info: ModelInformation, detections: Detections):
        def find_category_id_by_name(categories: List[Category], category_name: str):
            category_id = [category.id for category in categories if category.name == category_name]
            return category_id[0] if category_id else ''

        for box_detection in detections.box_detections:
            category_name = box_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            box_detection.category_id = category_id
        for point_detection in detections.point_detections:
            category_name = point_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            point_detection.category_id = category_id
        for segmentation_detection in detections.segmentation_detections:
            category_name = segmentation_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            segmentation_detection.category_id = category_id
        return detections


@contextlib.contextmanager
def pushd(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)
