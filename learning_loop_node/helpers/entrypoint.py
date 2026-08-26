"""The boilerplate every node's ``main.py`` repeats.

A node entry point always does the same four things: read a handful of settings, build the
logic object, construct the node, and hand it to uvicorn. Only the middle two are the node's
own, so :func:`node_parser` and :func:`run_node` cover the rest.

Settings come from a flag *or* an environment variable, because a node is configured on the
command line while developing and through the container environment in deployment. One
declaration gives both: ``--conf-threshold`` reads ``CONF_THRESHOLD``.

``--host`` and ``--port`` are the exception — they read ``NODE_HOST`` and ``NODE_PORT`` rather
than the names their flags imply, because the bare ``HOST`` already means *the loop's address*
and a node adopting it would hand it to uvicorn and fail to bind.
"""

import logging
import os
from argparse import Action, Namespace

import configargparse
import uvicorn


def node_parser(*, description: str, legacy_env_prefix: str = '') -> configargparse.ArgumentParser:
    """Build the parser for a node, pre-loaded with the settings every node has.

    :param legacy_env_prefix: A prefix an earlier version of this node required, e.g.
        ``'MY_DETECTOR_'``. Prefixed names are still honoured, with a warning, so a
        deployment keeps working until it is updated. Leave empty for a node that has always
        read unprefixed names.
    """
    parser = _NodeArgumentParser(description=description, legacy_env_prefix=legacy_env_prefix)
    parser.add_argument('--host', default='0.0.0.0', env_var='NODE_HOST',
                        help='Host interface to bind to')
    parser.add_argument('--port', type=int, default=80, env_var='NODE_PORT', help='Port to bind to')
    return parser


def run_node(app: str, args: Namespace) -> None:
    """Serve the node.

    :param app: Import string of the node object, conventionally ``'main:node'``.
    """
    reload = os.getenv('UVICORN_RELOAD', 'FALSE').lower() in ('true', '1')
    logging.info('Uvicorn reload is set to: %s', reload)
    uvicorn.run(app, host=args.host, port=args.port, lifespan='on', reload=reload)


class _NodeArgumentParser(configargparse.ArgumentParser):
    """Parser whose every setting is also an environment variable named after its flag."""

    def __init__(self, *, description: str, legacy_env_prefix: str) -> None:
        super().__init__(description=description)
        self.legacy_env_prefix = legacy_env_prefix

    def add_argument(self, *args, **kwargs) -> Action:  # type: ignore[override]
        action = super().add_argument(*args, **kwargs)
        if getattr(action, 'env_var', None) is None and action.dest != 'help':
            action.env_var = action.dest.upper()
        return action

    def parse_args(self, *args, **kwargs) -> Namespace:  # type: ignore[override]
        self._adopt_legacy_env_vars()
        return super().parse_args(*args, **kwargs)

    def _adopt_legacy_env_vars(self) -> None:
        if not self.legacy_env_prefix:
            return
        for action in self._actions:
            name = getattr(action, 'env_var', None)
            legacy = self.legacy_env_prefix + name if name else None
            if not legacy or name in os.environ or legacy not in os.environ:
                continue
            os.environ[name] = os.environ[legacy]
            logging.warning('%s is deprecated and will stop being read; set %s instead', legacy, name)
