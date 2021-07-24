# Learning Loop Node

This Python library helps you to write your own Detection Nodes, Training Nodes and Converter Nodes for the Zauberzeug Learning Loop.

## General Usage

You can configure connection to the learning loop by specifying the following environment variables before starting:

- HOST=learning-loop.ai
- ORGANIZATION=<your organization>
- PROJECT=<your project>
- USERNAME=<your username>
- PASSWORD=<your password>

## Detector Node

## Trainer Node
  
- if the command line tool "jpeginfo" is installed, the downloader will drop corrupted images automatically

## Converter Node

A Conveter Node converts models from one format into another.

### How to test the operability?

Assumend there is a Converter Node which converts models of format 'format_a' into 'format_b'.
Upload a model with
`curl --request POST -F 'files=@my_model.zip' https://learning-loop.ai/api/zauberzeug/projects/demo/format_a`
The model should now be available for the format 'format_a'
`curl "https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_a"`

```
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

```
{
  "models": []
}
```

Connect the Node to the learning loop by simply starting the container.
After a short time the converted Model should be available as well.
`curl https://learning-loop.ai/api/zauberzeug/projects/demo/models?format=format_b`

```
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
