import asyncio
import contextlib
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

try:
    from typing import Literal
except ImportError:  # Python <= 3.8
    from typing_extensions import Literal  # type: ignore

import socketio
from dacite import from_dict
from fastapi.encoders import jsonable_encoder
from socketio import AsyncClient

from ..data_classes import (
    AboutResponse,
    Category,
    Context,
    DetectorStatus,
    ImageMetadata,
    ImagesMetadata,
    ModelInformation,
    ModelVersionResponse,
    Shape,
)
from ..data_exchanger import DataExchanger, DownloadError
from ..enums import OperationMode, VersionMode
from ..globals import GLOBALS
from ..helpers import background_tasks, environment_reader, run
from ..helpers.misc import numpy_image_from_dict
from ..node import Node
from .detector_logic import DetectorLogic
from .exceptions import NodeNeedsRestartError
from .inbox_filter.relevance_filter import RelevanceFilter
from .outbox import Outbox
from .rest import about as rest_about
from .rest import backdoor_controls
from .rest import detect as rest_detect
from .rest import model_version_control as rest_version_control
from .rest import operation_mode as rest_mode
from .rest import outbox_mode as rest_outbox_mode
from .rest import upload as rest_upload


class DetectorNode(Node):

    def __init__(self, name: str, detector: DetectorLogic, uuid: Optional[str] = None, use_backdoor_controls: bool = False) -> None:
        super().__init__(name, uuid=uuid, node_type='detector', needs_login=False, needs_sio=False)
        self.detector_logic = detector
        self.organization = environment_reader.organization()
        self.project = environment_reader.project()
        assert self.organization and self.project, 'Detector node needs an organization and an project'
        self.log.info('Using %s/%s', self.organization, self.project)
        self.operation_mode: OperationMode = OperationMode.Startup
        self.connected_clients: List[str] = []

        self.detection_lock = asyncio.Lock()

        self.outbox: Outbox = Outbox()
        self.data_exchanger = DataExchanger(
            Context(organization=self.organization, project=self.project),
            self.loop_communicator)

        self.relevance_filter: RelevanceFilter = RelevanceFilter(self.outbox)

        # NOTE: version_control controls the behavior of the detector node.
        # FollowLoop: the detector node will follow the loop and update the model if necessary
        # SpecificVersion: the detector node will update to a specific version, set via the /model_version endpoint
        # Pause: the detector node will not update the model
        self.version_control: VersionMode = VersionMode.Pause if os.environ.get(
            'VERSION_CONTROL_DEFAULT', 'follow_loop').lower() == 'pause' else VersionMode.FollowLoop
        self.target_model: Optional[ModelInformation] = None
        self.loop_deployment_target: Optional[ModelInformation] = None

        self._regular_status_sync_cycles: int = int(os.environ.get('SYNC_CYCLES', '6'))
        """sync status every 6 cycles (6*10s = 1min)"""
        self._repeat_cycles_to_next_sync: int = 0

        self.include_router(rest_detect.router, tags=["detect"])
        self.include_router(rest_upload.router, prefix="")
        self.include_router(rest_mode.router, tags=["operation_mode"])
        self.include_router(rest_about.router, tags=["about"])
        self.include_router(rest_outbox_mode.router, tags=["outbox_mode"])
        self.include_router(rest_version_control.router, tags=["model_version"])

        if use_backdoor_controls or os.environ.get('USE_BACKDOOR_CONTROLS', '0').lower() in ('1', 'true'):
            self.include_router(backdoor_controls.router)

        self._setup_sio_server()

    def get_about_response(self) -> AboutResponse:
        return AboutResponse(
            operation_mode=self.operation_mode.value,
            state=self.status.state,
            model_info=self.detector_logic.model_info,  # pylint: disable=protected-access
            target_model=self.target_model.version if self.target_model else None,
            version_control=self.version_control.value
        )

    def get_model_version_response(self) -> ModelVersionResponse:
        current_version = self.detector_logic.model_info.version if self.detector_logic.model_info is not None else 'None'  # pylint: disable=protected-access
        target_version = self.target_model.version if self.target_model is not None else 'None'
        loop_version = self.loop_deployment_target.version if self.loop_deployment_target is not None else 'None'

        local_versions: list[str] = []
        models_path = os.path.join(GLOBALS.data_folder, 'models')
        local_models = os.listdir(models_path) if os.path.exists(models_path) else []
        for model in local_models:
            if model.replace('.', '').isdigit():
                local_versions.append(model)

        return ModelVersionResponse(
            current_version=current_version,
            target_version=target_version,
            loop_version=loop_version,
            local_versions=local_versions,
            version_control=self.version_control.value,
        )

    async def set_model_version_mode(self, version_control_mode: str) -> None:

        self.log.info('Setting model version mode to %s', version_control_mode)

        if version_control_mode == 'follow_loop':
            self.version_control = VersionMode.FollowLoop
        elif version_control_mode == 'pause':
            self.version_control = VersionMode.Pause
        else:
            self.version_control = VersionMode.SpecificVersion
            if not version_control_mode or not version_control_mode.replace('.', '').isdigit():
                raise Exception('Invalid version number')
            target_version = version_control_mode

            if self.target_model is not None and self.target_model.version == target_version:
                return

            # Fetch the model uuid by version from the loop
            uri = f'/{self.organization}/projects/{self.project}/models'
            response = await self.loop_communicator.get(uri)
            if response.status_code != 200:
                self.version_control = VersionMode.Pause
                raise Exception('Failed to load models from learning loop')

            models = response.json()['models']
            models_with_target_version = [m for m in models if m['version'] == target_version]
            if len(models_with_target_version) == 0:
                self.version_control = VersionMode.Pause
                raise Exception(f'No Model with version {target_version}')
            if len(models_with_target_version) > 1:
                self.version_control = VersionMode.Pause
                raise Exception(f'Multiple models with version {target_version}')

            model_id = models_with_target_version[0]['id']
            model_host = models_with_target_version[0].get('host', 'unknown')

            self.target_model = ModelInformation(organization=self.organization, project=self.project,
                                                 host=model_host, categories=[],
                                                 id=model_id,
                                                 version=target_version)

    async def soft_reload(self) -> None:
        # simulate init
        self.organization = environment_reader.organization()
        self.project = environment_reader.project()
        self.operation_mode = OperationMode.Startup
        self.connected_clients = []
        self.data_exchanger = DataExchanger(
            Context(organization=self.organization, project=self.project),
            self.loop_communicator)
        self.relevance_filter = RelevanceFilter(self.outbox)
        self.version_control = VersionMode.Pause if os.environ.get(
            'VERSION_CONTROL_DEFAULT', 'follow_loop').lower() == 'pause' else VersionMode.FollowLoop
        self.target_model = None
        # self.setup_sio_server()

        # simulate super().startup
        await self.loop_communicator.backend_ready()
        # await self.loop_communicator.ensure_login()
        self.set_skip_repeat_loop(False)
        self.socket_connection_broken = True
        await self.on_startup()

        # simulate startup
        await self.detector_logic.soft_reload()
        await self._load_model_info_and_init_model()
        self.operation_mode = OperationMode.Idle

    async def on_startup(self) -> None:
        try:
            self.outbox.ensure_continuous_upload()
            await self._load_model_info_and_init_model()
        except Exception:
            self.log.exception("error during 'startup'")
        self.operation_mode = OperationMode.Idle

    async def on_shutdown(self) -> None:
        try:
            await self.outbox.ensure_continuous_upload_stopped()
            for sid in self.connected_clients:
                # pylint: disable=no-member
                await self.sio.disconnect(sid)  # type:ignore
        except Exception:
            self.log.exception("error during 'shutdown'")

    def _setup_sio_server(self) -> None:
        """The DetectorNode acts as a SocketIO server. This method sets up the server and defines the event handlers."""
        # pylint: disable=unused-argument

        # Initialize the Socket.IO server with 20MB buffer size
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            max_http_buffer_size=2e7,  # 20MB
        )
        # Initialize and mount the ASGI app
        self.sio_app = socketio.ASGIApp(self.sio, socketio_path='/socket.io')
        self.mount('/ws', self.sio_app)
        # Register event handlers

        self.log.info('>>>>>>>>>>>>>>>>>>>>>>> Setting up the SIO server')

        @self.sio.event
        async def detect(sid, data: Dict) -> Dict:
            """Detect objects in a single image sent via SocketIO.

            The data dict has the following schema:
            - image: The image data as dictionary:
              - bytes: bytes of the ndarray
              - dtype: data type of the ndarray
              - shape: shape of the ndarray
            - camera_id: Optional camera ID
            - tags: Optional list of tags
            - source: Optional source string
            - autoupload: Optional 'filtered', 'all' or 'disabled' (default: 'filtered')
            - creation_date: Optional creation date in isoformat string
            """
            try:
                image = numpy_image_from_dict(data['image'])
            except Exception:
                self.log.exception('could not parse image from socketio')
                return {'error': 'could not parse image from data'}

            try:
                det = await self.get_detections(
                    image=image,
                    camera_id=data.get('camera_id', None),
                    tags=data.get('tags', []),
                    source=data.get('source', None),
                    autoupload=data.get('autoupload', 'filtered'),
                    creation_date=data.get('creation_date', None)
                )
                if det is None:
                    return {'error': 'no model loaded'}
                detection_dict = jsonable_encoder(asdict(det))
                return detection_dict
            except Exception as e:
                self.log.exception('could not detect via socketio')
                # with open('/tmp/bad_img_from_socket_io.jpg', 'wb') as f:
                #     f.write(data['image'])
                return {'error': str(e)}

        @self.sio.event
        async def batch_detect(sid, data: Dict) -> Dict:
            """
            Detect objects in a batch of images sent via SocketIO.

            Data dict follows the schema of the detect endpoint, 
            but 'images' is a list of image dicts.
            """
            try:
                images_data = data['images']
                images = [numpy_image_from_dict(image) for image in images_data]
            except Exception:
                self.log.exception('could not parse images from socketio')
                return {'error': 'could not parse images from data'}

            try:
                det = await self.get_batch_detections(
                    images=images,
                    tags=data.get('tags', []),
                    camera_id=data.get('camera_id', None),
                    source=data.get('source', None),
                    autoupload=data.get('autoupload', 'filtered'),
                    creation_date=data.get('creation_date', None)
                )
                if det is None:
                    return {'error': 'no model loaded'}
                detection_dict = jsonable_encoder(asdict(det))
                return detection_dict
            except Exception as e:
                self.log.exception('could not detect via socketio')
                # with open('/tmp/bad_img_from_socket_io.jpg', 'wb') as f:
                #     f.write(data['image'])
                return {'error': str(e)}

        @self.sio.event
        async def info(sid) -> Dict:
            if self.detector_logic.model_info is not None:
                return asdict(self.detector_logic.model_info)
            return {"status": "No model loaded"}

        @self.sio.event
        async def about(sid) -> Dict:
            return asdict(self.get_about_response())

        @self.sio.event
        async def get_model_version(sid) -> Dict:
            return asdict(self.get_model_version_response())

        @self.sio.event
        async def set_model_version_mode(sid, data: str) -> Dict:
            try:
                await self.set_model_version_mode(data)
                return {"status": "OK"}
            except Exception as e:
                return {'error': str(e)}

        @self.sio.event
        async def get_outbox_mode(sid) -> Dict:
            return {'outbox_mode': self.outbox.get_mode().value}

        @self.sio.event
        async def set_outbox_mode(sid, data: str) -> Dict:
            try:
                await self.outbox.set_mode(data)
                return {"status": "OK"}
            except Exception as e:
                return {'error': str(e)}

        @self.sio.event
        async def upload(sid, data: Dict) -> Dict:
            """Upload a single image with metadata to the learning loop.

            The data dict must contain:
            - image: The image data as dictionary with the following keys:
                - bytes: bytes of the ndarray (retrieved via `ndarray.tobytes(order='C')`)
                - dtype: data type of the ndarray as string (e.g. `uint8`, `float32`, etc.)
                - shape: shape of the ndarray as tuple of ints (e.g. `(480, 640, 3)`)
            - metadata: The metadata for the image (optional)
            - upload_priority: Whether to upload with priority (optional)
            """
            self.log.debug('Processing upload via socketio.')

            metadata = data.get('metadata', None)
            if metadata:
                try:
                    image_metadata = from_dict(data_class=ImageMetadata, data=metadata)
                except Exception as e:
                    self.log.exception('could not parse detections')
                    return {'error': str(e)}
                if self.detector_logic.model_info is not None:
                    image_metadata = self.add_category_id_to_detections(self.detector_logic.model_info, image_metadata)
            else:
                image_metadata = ImageMetadata()

            try:
                image = numpy_image_from_dict(data['image'])
            except Exception:
                self.log.exception('could not parse image from socketio')
                return {'error': 'could not parse image from data'}

            try:
                await self.upload_images(
                    images=[image],
                    images_metadata=ImagesMetadata(items=[image_metadata]) if metadata else None,
                    upload_priority=data.get('upload_priority', False)
                )
            except Exception as e:
                self.log.exception('could not upload via socketio')
                return {'error': str(e)}
            return {'status': 'OK'}

        @self.sio.event
        def connect(sid, environ, auth) -> None:
            self.connected_clients.append(sid)

# ================================== Repeat Cycle, sync and model updates ==================================

    async def on_repeat(self) -> None:
        """Implementation of the repeat cycle. This method is called every 10 seconds.
        To avoid too many requests, the status is only synced every 6 cycles (1 minute)."""
        try:
            self._repeat_cycles_to_next_sync -= 1
            if self._repeat_cycles_to_next_sync <= 0:
                self._repeat_cycles_to_next_sync = self._regular_status_sync_cycles
                await self._sync_status_with_loop()
            await self._update_model_if_required()
        except Exception:
            self.log.exception("error during '_check_for_update'")

    async def _sync_status_with_loop(self) -> None:
        """Sync status of the detector with the Learning Loop."""

        if self.detector_logic.model_info is not None:
            current_model = self.detector_logic.model_info.version
        else:
            current_model = None

        target_model_version = self.target_model.version if self.target_model else None

        status = DetectorStatus(
            uuid=self.uuid,
            name=self.name,
            state=self.status.state,
            errors=self.status.errors,
            uptime=int((datetime.now() - self.startup_datetime).total_seconds()),
            operation_mode=self.operation_mode,
            current_model=current_model,
            target_model=target_model_version,
            model_format=self.detector_logic.model_format,
        )

        self.log_status_on_change(status.state, status)
        response = None

        try:
            response = await self.loop_communicator.post(
                f'/{self.organization}/projects/{self.project}/detectors', json=jsonable_encoder(asdict(status)))
        except Exception:
            self.log.warning('Exception while trying to sync status with loop')

        if not response or not response.is_success:
            self.log.warning('Status update failed. Response: "%s"', response)

    async def _update_model_if_required(self) -> None:
        """Check if a new model is available and update if necessary.
        The Learning Loop will respond with the model info of the deployment target.
        If version_control is set to FollowLoop or the chosen target model is not used, 
        the detector will update the target_model."""
        try:
            if self.operation_mode != OperationMode.Idle:
                self.log.debug('not checking for updates; operation mode is %s', self.operation_mode)
                return

            await self._check_for_new_deployment_target()

            self.status.reset_error('update_model')
            if self.target_model is None:
                self.log.debug('not running any updates; target model is None')
                return

            current_version = self.detector_logic.model_info.version \
                if self.detector_logic.model_info is not None else None

            if current_version != self.target_model.version:
                self.log.info('Updating model from %s to %s',
                              current_version or "-", self.target_model.version)
                await self._update_model(self.target_model)

        except Exception as e:
            self.log.exception('check_for_update failed')
            msg = e.cause if isinstance(e, DownloadError) else str(e)
            self.status.set_error('update_model', f'Could not update model: {msg}')
            await self._sync_status_with_loop()

    async def _check_for_new_deployment_target(self) -> None:
        """Ask the learning loop for the current deployment target and update self.loop_deployment_target.
        If version_control is set to FollowLoop, also update target_model."""
        try:
            response = await self.loop_communicator.get(
                f'/{self.organization}/projects/{self.project}/deployment/target')
        except Exception:
            self.log.warning('Exception while trying to check for new deployment target')
            return

        if response.status_code != 200:
            self.log.warning('Failed to check for new deployment target: %s', str(response))
            return

        response_data = response.json()

        deployment_target_uuid = response_data['model_uuid']
        deployment_target_version = response_data['version']
        self.loop_deployment_target = ModelInformation(organization=self.organization, project=self.project,
                                                       host="", categories=[],
                                                       id=deployment_target_uuid,
                                                       version=deployment_target_version)

        if (self.version_control == VersionMode.FollowLoop and
                self.target_model != self.loop_deployment_target):
            previous_version = self.target_model.version if self.target_model else None
            self.target_model = self.loop_deployment_target
            self.log.info('Deployment target changed from %s to %s',
                          previous_version, self.target_model.version)

    async def _update_model(self, target_model: ModelInformation) -> None:
        """Download and install the target model.
        On failure, the target_model will be set to None which will trigger a retry on the next check."""

        with step_into(GLOBALS.data_folder):
            target_model_folder = f'models/{target_model.version}'
            if os.path.exists(target_model_folder) and len(os.listdir(target_model_folder)) > 0:
                self.log.info('No need to download model. %s (already exists)', target_model.version)
            else:
                os.makedirs(target_model_folder, exist_ok=True)
                try:
                    await self.data_exchanger.download_model(target_model_folder,
                                                             Context(organization=self.organization,
                                                                     project=self.project),
                                                             target_model.id, self.detector_logic.model_format)
                    self.log.info('Downloaded model %s', target_model.version)
                except Exception:
                    self.log.exception('Could not download model %s', target_model.version)
                    shutil.rmtree(target_model_folder, ignore_errors=True)
                    self.target_model = None
                    return

            model_symlink = 'model'
            try:
                os.unlink(model_symlink)
                os.remove(model_symlink)
            except Exception:
                pass
            os.symlink(target_model_folder, model_symlink)
            self.log.info('Updated symlink for model to %s', os.readlink(model_symlink))

            try:
                await self._load_model_info_and_init_model()
            except NodeNeedsRestartError:
                self.log.error('Node needs restart')
                sys.exit(0)
            except Exception:
                self.log.exception('Could not load model, will retry download on next check')
                shutil.rmtree(target_model_folder, ignore_errors=True)
                self.target_model = None
                return

            await self._sync_status_with_loop()
            # self.reload(reason='new model installed')

    async def _load_model_info_and_init_model(self) -> None:
        async with self.detection_lock:
            self.detector_logic.load_model_info_and_init_model()

# ================================== API Implementations ==================================

    async def set_operation_mode(self, mode: OperationMode):
        self.operation_mode = mode
        await self._sync_status_with_loop()

    def reload(self, reason: str):
        """provide a cause for the reload"""

        self.log.info('########## reloading app because %s', reason)
        if os.path.isfile('/app/app_code/restart/restart.py'):
            subprocess.call(['touch', '/app/app_code/restart/restart.py'])
        elif os.path.isfile('/app/main.py'):
            subprocess.call(['touch', '/app/main.py'])
        elif os.path.isfile('/main.py'):
            subprocess.call(['touch', '/main.py'])
        else:
            self.log.error('could not reload app')

    async def get_detections(self,
                             image: np.ndarray,
                             tags: List[str],
                             *,
                             camera_id: Optional[str] = None,
                             source: Optional[str] = None,
                             autoupload: Literal['filtered', 'all', 'disabled'],
                             creation_date: Optional[str] = None) -> ImageMetadata:
        """
        Main processing function for the detector node.

        Used when an image is received via REST or SocketIO.
        This function infers the detections from the image, 
        cares about uploading to the loop and returns the detections as ImageMetadata object.
        """

        async with self.detection_lock:
            metadata = await run.io_bound(self.detector_logic.evaluate, image)

        metadata.tags.extend(tags)
        metadata.source = source
        metadata.created = creation_date

        fix_shape_detections(metadata)
        n_bo, n_cl = len(metadata.box_detections), len(metadata.classification_detections)
        n_po, n_se = len(metadata.point_detections), len(metadata.segmentation_detections)
        self.log.debug('Detected: %d boxes, %d points, %d segs, %d classes', n_bo, n_po, n_se, n_cl)

        if autoupload == 'filtered':
            background_tasks.create(self.relevance_filter.may_upload_detections(metadata, camera_id, image))
        elif autoupload == 'all':
            background_tasks.create(self.outbox.save(image, metadata))
        elif autoupload == 'disabled':
            pass
        else:
            self.log.error('unknown autoupload value %s', autoupload)
        return metadata

    async def get_batch_detections(self,
                                   images: List[np.ndarray],
                                   tags: List[str],
                                   *,
                                   camera_id: Optional[str] = None,
                                   source: Optional[str] = None,
                                   autoupload: str = 'filtered',
                                   creation_date: Optional[str] = None) -> ImagesMetadata:
        """
        Processing function for the detector node when a a batch inference is requested via SocketIO.

        This function infers the detections from all images, 
        cares about uploading to the loop and returns the detections as a list of ImageMetadata.
        """

        async with self.detection_lock:
            all_detections = await run.io_bound(self.detector_logic.batch_evaluate, images)

        for metadata in all_detections.items:
            metadata.tags.extend(tags)
            metadata.source = source
            metadata.created = creation_date

        for detections, image in zip(all_detections.items, images):
            fix_shape_detections(detections)
            n_bo, n_cl = len(detections.box_detections), len(detections.classification_detections)
            n_po, n_se = len(detections.point_detections), len(detections.segmentation_detections)
            self.log.debug('Detected: %d boxes, %d points, %d segs, %d classes', n_bo, n_po, n_se, n_cl)

            if autoupload == 'filtered':
                background_tasks.create(self.relevance_filter.may_upload_detections(detections, camera_id, image))
            elif autoupload == 'all':
                background_tasks.create(self.outbox.save(image, detections))
            elif autoupload == 'disabled':
                pass
            else:
                self.log.error('unknown autoupload value %s', autoupload)
        return all_detections

    async def upload_images(
            self, *,
            images: List[np.ndarray],
            images_metadata: Optional[ImagesMetadata] = None,
            upload_priority: bool = False
    ) -> None:
        """Save images to the outbox using an asyncio executor.
        Used by SIO and REST upload endpoints.

        :param images: List of images to upload
        :param images_metadata: Optional metadata for all images
        :param upload_priority: Whether to upload the images with priority
        :raises ValueError: If the number of images and number of metadata items do not match
        """

        if images_metadata and len(images_metadata.items) != len(images):
            raise ValueError('Number of images and number of metadata items do not match')

        for i, image in enumerate(images):
            image_metadata = images_metadata.items[i] if images_metadata else ImageMetadata()
            image_metadata.tags.append('picked_by_system')
            await self.outbox.save(image, image_metadata, upload_priority)

    def add_category_id_to_detections(self, model_info: ModelInformation, image_metadata: ImageMetadata):
        def find_category_id_by_name(categories: List[Category], category_name: str):
            category_id = [category.id for category in categories if category.name == category_name]
            return category_id[0] if category_id else ''

        for box_detection in image_metadata.box_detections:
            category_name = box_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            box_detection.category_id = category_id
        for point_detection in image_metadata.point_detections:
            category_name = point_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            point_detection.category_id = category_id
        for segmentation_detection in image_metadata.segmentation_detections:
            category_name = segmentation_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            segmentation_detection.category_id = category_id
        for classification_detection in image_metadata.classification_detections:
            category_name = classification_detection.category_name
            category_id = find_category_id_by_name(model_info.categories, category_name)
            classification_detection.category_id = category_id
        return image_metadata

    def register_sio_events(self, sio_client: AsyncClient):
        pass


@contextlib.contextmanager
def step_into(new_dir):
    previous_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(previous_dir)


def fix_shape_detections(metadata: ImageMetadata):
    # TODO This is a quick fix.. check how loop upload detections deals with this
    for seg_detection in metadata.segmentation_detections:
        if isinstance(seg_detection.shape, Shape):
            points = ','.join([str(value) for p in seg_detection.shape.points for _,
                               value in asdict(p).items()])
            seg_detection.shape = points
