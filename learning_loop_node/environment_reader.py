import os


def organization(default: str = None):
    return os.environ.get('LOOP_ORGANIZATION', default) or os.environ.get('ORGANIZATION', default)


def project(default: str = None):
    return os.environ.get('LOOP_PROJECT', default) or os.environ.get('PROJECT', default)
