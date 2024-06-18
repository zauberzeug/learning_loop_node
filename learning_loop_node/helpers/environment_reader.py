import logging
import os
from typing import List, Optional


# TODO ignore_errors should default to False, but maybe some tests rely on this behavior
def read_from_env(possible_names: List[str], ignore_errors: bool = True) -> Optional[str]:
    values = [os.environ.get(name, None) for name in possible_names]
    values = list(filter(None, values))

    # Possible error: no values are set
    if not values:
        if ignore_errors:
            logging.warning(f'no environment variable set for {possible_names}')
            return None
        raise ValueError(f'no environment variable set for {possible_names}')

    # Possible error: multiple values are not None and not equal
    if len(values) > 1 and len(set(values)) > 1:
        if ignore_errors:
            logging.warning(f'different environment variables set for {possible_names}: {values}')
            return None
        raise ValueError(f'different environment variables set for {possible_names}: {values}')

    return values[0]


def organization(default: str = '') -> str:
    return read_from_env(['LOOP_ORGANIZATION', 'ORGANIZATION']) or default


def project(default: str = '') -> str:
    return read_from_env(['LOOP_PROJECT', 'PROJECT']) or default


def username(default: str = '') -> str:
    return read_from_env(['LOOP_USERNAME', 'USERNAME']) or default


def password(default: str = '') -> str:
    return read_from_env(['LOOP_PASSWORD', 'PASSWORD']) or default


def host(default: str = '') -> str:
    return read_from_env(['LOOP_HOST', 'HOST']) or default


def ssl_certificate_path(default: str = '') -> str:
    return read_from_env(['LOOP_SSL_CERT_PATH',]) or default
