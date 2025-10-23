# Learning Loop Node

This Python library helps to write Nodes that interact with the Zauberzeug Learning Loop. There are 4 types of Nodes:

| Type      | Purpose                                              |
| --------- | ---------------------------------------------------- |
| Trainer   | Runs training using the latest training data         |
| Detector  | Loads latest models from Loop and performs inference |
| Annotator | Used for custom annotation inside the Loop           |
| Converter | Converts between different model formats             |

## General Usage

To start a node you have to implement the logic by inheriting from the corresponding base logic class. We provide samples in the 'mock' folders and recommend to follow that scheme. A complete trainer and detector example can be found [here](https://github.com/zauberzeug/yolov5_node).

#### Environment variables

You can configure connection to our Learning Loop by specifying the following environment variables before starting:

| Name                     | Alias        | Purpose                                                      | Required by               |
| ------------------------ | ------------ | ------------------------------------------------------------ | ------------------------- |
| LOOP_HOST                | HOST         | Learning Loop address (e.g. learning-loop.ai)                | all                       |
| LOOP_USERNAME            | USERNAME     | Learning Loop user name                                      | all besides Detector      |
| LOOP_PASSWORD            | PASSWORD     | Learning Loop password                                       | all besides Detector      |
| LOOP_SSL_CERT_PATH       | -            | Path to the SSL certificate                                  | all (opt.)                |
| LOOP_ORGANIZATION        | ORGANIZATION | Organization ID                                              | Detector                  |
| LOOP_PROJECT             | PROJECT      | Project ID                                                   | Detector (opt.)           |
| MIN_UNCERTAIN_THRESHOLD  | -            | smallest confidence (float) at which auto-upload will happen | Detector (opt.)           |
| MAX_UNCERTAIN_THRESHOLD  | -            | largest confidence (float) at which auto-upload will happen  | Detector (opt.)           |
| INFERENCE_BATCH_SIZE     | -            | Batch size of trainer when calculating detections            | Trainer (opt.)            |
| RESTART_AFTER_TRAINING   | -            | Restart the trainer after training (set to 1)                | Trainer (opt.)            |
| KEEP_OLD_TRAININGS       | -            | Do not delete old trainings (set to 1)                       | Trainer (opt.)            |
| TRAINER_IDLE_TIMEOUT_SEC | -            | Automatically shutdown trainer after timeout (in seconds)    | Trainer (opt.)            |
| USE_BACKDOOR_CONTROLS    | -            | Always enable backdoor controls (set to 1)                   | Trainer / Detector (opt.) |

Note that organization and project IDs are always lower case and may differ from the names in the Learning Loop which can have uppercase letters.

#### Testing

We use github actions for CI. Tests can also be executed locally by running
`LOOP_HOST=XXXXXXXX LOOP_USERNAME=XXXXXXXX LOOP_PASSWORD=XXXXXXXX python -m pytest -v`  
from learning_loop_node/learning_loop_node

## Detector Node

Detector Nodes are normally deployed on edge devices like robots or machinery but can also run in the cloud to provide backend services for an app or similar. These nodes register themself at the Learning Loop. They provide REST and Socket.io APIs to run inference on images. The processed images can automatically be used for active learning: e.g. uncertain predictions will be send to the Learning Loop.

### Inference API

Images can be send to the detector node via socketio or rest.
Via **REST** you may provide the following parameters:

- `camera_id`: a camera identifier (string) used to improve the autoupload filtering
- `tags`: comma separated list of tags to add to the image in the learning loop to add to the image in the learning loop
- `source`: optional source identifier (str) for the image (e.g. a robot id)
- `autoupload`: configures auto-submission to the learning loop; `filtered` (default), `all`, `disabled`
- `creation_date`: optional creation date (str) for the image in isoformat (e.g. `2023-01-30T12:34:56`)

Example usage:

`curl --request POST -F 'file=@test.jpg' -H 'autoupload: all' -H 'camera_id: front_cam' localhost:8004/detect`

To use the **SocketIO** inference EPs, the caller needs to connect to the detector node's SocketIO server and emit the `detect` or `batch_detect` event with the image data and image metadata. The `detect` endpoint receives a dictionary, with the following entries:

- `image`: The image data as dictionary with the following keys:
  - `bytes`: bytes of the ndarray (retrieved via `ndarray.tobytes(order='C')`)
  - `dtype`: data type of the ndarray as string (e.g. `uint8`, `float32`, etc.)
  - `shape`: shape of the ndarray as tuple of ints (e.g. `(480, 640, 3)`)
- `camera_id`: optional camera identifier (string) used to improve the autoupload filtering
- `tags`: optional list of tags to add to the image in the learning loop
- `source`: optional source string
- `autoupload`: configures auto-submission to the learning loop; `filtered` (default), `all`, `disabled`
- `creation_date`: optional creation date (str) for the image in isoformat (e.g. `2023-01-30T12:34:56`)

The `batch_detect` endpoint receives a dictionary, with the same entries as the `detect` endpoint, except that the `image` entry is replaced by:

- `images`: List of image data dictionaries, each with the same structure as the `image` entry in the `detect` endpoint

Example code can be found [in the rosys implementation](https://github.com/zauberzeug/rosys/blob/main/rosys/vision/detector_hardware.py).

### Upload API

The detector has a **REST** endpoint to upload images (and detections) to the Learning Loop. The endpoint takes a POST request with one or multiple images. The images are expected to be in jpg format. The following optional parameters may be set via headers:

- `source`: optional source identifier (str) for the image (e.g. a robot id)
- `creation_date`: optional creation date (str) for the image in isoformat (e.g. `2023-01-30T12:34:56`)
- `upload_priority`: A boolean flag to prioritize the upload (defaults to False)

Example:

`curl -X POST -F 'files=@test.jpg' "http://localhost:/upload"`

The detector also has a **SocketIO** upload endpoint that can be used to upload images and detections to the learning loop. The function receives a dictionary, with the following entries:

- `image`: the image data as dictionary with the following keys:
  - `bytes`: bytes of the ndarray (retrieved via `ndarray.tobytes(order='C')`)
  - `dtype`: data type of the ndarray as string (e.g. `uint8`, `float32`, etc.)
  - `shape`: shape of the ndarray as tuple of ints (e.g. `(480, 640, 3)`)
- `metadata`: a dictionary representing the image metadata. If metadata contains detections and/or annotations, UUIDs for the classes are automatically determined based on the category names. Metadata should follow the schema of the `ImageMetadata` data class.
- `upload_priority`: Optional boolean flag to prioritize the upload (defaults to False)

The endpoint returns None if the upload was successful and an error message otherwise.

For both ways to upload an image, the tag `picked_by_system` is automatically added to the image metadata.

### Changing the model versioning mode

The detector can be configured to one of the following behaviors:

- use a specific model version
- automatically update the model version according to the learning loop deployment target
- pause the model updates and use the version that was last loaded

The model versioning configuration can be accessed/changed via a REST endpoint. Example Usage:

- Fetch the current model versioning configuration: `curl http://localhost/model_version`
- Configure the detector to use a specific model version: `curl -X PUT -d "1.0" http://localhost/model_version`
- Configure the detector to automatically update the model version: `curl -X PUT -d "follow_loop" http://localhost/model_version`
- Pause the model updates: `curl -X PUT -d "pause" http://localhost/model_version`

Note that the configuration is not persistent, however, the default behavior on startup can be configured via the environment variable `VERSION_CONTROL_DEFAULT`.
If the environment variable is set to `VERSION_CONTROL_DEFAULT=PAUSE`, the detector will pause the model updates on startup. Otherwise, the detector will automatically follow the loop deployment target.

The model versioning configuration can also be changed via a socketio event:

- Configure the detector to use a specific model version: `sio.emit('set_model_version_mode', '1.0')`
- Configure the detector to automatically update the model version: `sio.emit('set_model_version_mode', 'follow_loop')`
- Pause the model updates: `sio.emit('set_model_version_mode', 'pause')`

There is also a GET endpoint to fetch the current model versioning configuration:
`sio.emit('get_model_version')` or `curl http://localhost/model_version`

### Changing the outbox mode

If the autoupload is set to `all` or `filtered` (selected) images and the corresponding detections are saved on HDD (the outbox). A background thread will upload the images and detections to the Learning Loop. The outbox is located in the `outbox` folder in the root directory of the node. The outbox can be cleared by deleting the files in the folder.

The continuous upload can be stopped/started via a REST enpoint:

Example Usage:

- Enable upload: `curl -X PUT -d "continuous_upload" http://localhost/outbox_mode`
- Disable upload: `curl -X PUT -d "stopped" http://localhost/outbox_mode`

The current state can be queried via a GET request:
`curl http://localhost/outbox_mode`

Alternatively, the outbox mode can be changed via a socketio event:

- Enable upload: `sio.emit('set_outbox_mode', 'continuous_upload')`
- Disable upload: `sio.emit('set_outbox_mode', 'stopped')`

The outbox mode can also be queried via:

- HTTP: `curl http://localhost/outbox_mode`
- SocketIO: `sio.emit('get_outbox_mode')`

## Trainer Node

Trainers fetch the images and anntoations from the Learning Loop to train new models.

- if the command line tool "jpeginfo" is installed, the downloader will drop corrupted images automatically

## Converter Node

A Conveter Node converts models from one format into another.

## Annotator Node

...

### Test operability

Assumend there is a Converter Node which converts models of format 'format_a' into 'format_b'.
Upload a model with
`curl --request POST -F 'files=@my_model.zip' https://learning-loop.ai/api/zauberzeug/projects/demo/format_a`
The model should now be available for the format 'format_a'
`curl "https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_a"`

```json
{
  "models": [
    {
    "id": "3c20d807-f71c-40dc-a996-8a8968aa5431",
    "version": "4.0",
    "formats": [
      "format_a"
    ],
    "created": "2021-06-01T06:28:21.289092",
    "comment": "uploaded at 2021-06-01 06:28:21.288442",
    ...
    }
  ]
}
```

but not in the format_b
`curl "https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_b"`

```json
{
  "models": []
}
```

Connect the Node to the Learning Loop by simply starting the container.
After a short time the converted model should be available as well.
`curl https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_b`

```json
{
  "models": [
  {
  "id": "3c20d807-f71c-40dc-a996-8a8968aa5431",
    "version": "4.0",
    "formats": [
      "format_a",
      "format_b",
    ],
    "created": "2021-06-01T06:28:21.289092",
    "comment": "uploaded at 2021-06-01 06:28:21.288442",
    ...
  }
  ]
}
```

## About Models (the currency between Nodes)

- Models are packed in zips and saved on the Learning Loop (one for each format)
- Nodes and users can upload and download models with which they want to work
- In each zip there is a file called `model.json` which contains the metadata to interpret the other files in the package
- for base models (pretrained models from external sources) no `model.json` has to be sent, ie. these models should simply be zipped in such a way that the respective trainer can work with them.
- the loop adds or corrects the following properties in the `model.json` after receiving; it also creates the file if it is missing:
  - `host`: uri to the loop
  - `organization`: the ID of the organization
  - `project`: the id of the project
  - `version`: the version number that the loop assigned for this model (e.g. 1.3)
  - `id`: the model UUID (currently not needed by anyone, since host, org, project, version clearly identify the model)
  - `format`: the format e.g. yolo, tkdnn, yolor etc.
- Nodes add properties to `model.json`, which contains all the information which are needed by subsequent nodes. These are typically the properties:
  - `resolution`: resolution in which the model expects images (as `int`, since the resolution is mostly square - later, ` resolution_x`` resolution_y ` would also be conceivable or `resolutions` to give a list of possible resolutions)
  - `categories`: list of categories with name, id, (later also type), in the order in which they are used by the model -- this is neccessary to be robust about renamings
