import logging
import os

logger = logging.getLogger(__name__)


# TODO ignore_errors should default to False, but maybe some tests rely on this behavior
def read_from_env(possible_names: list[str], ignore_errors: bool = True) -> str | None:
    """Read the first of ``possible_names`` that is set.

    :param possible_names: In order of preference; on a disagreement the first one set wins.
    :raises ValueError: If nothing is set or the values disagree, unless ``ignore_errors``.
    """
    values = [os.environ.get(name, None) for name in possible_names]
    values = list(filter(None, values))

    # Possible error: no values are set
    if not values:
        if ignore_errors:
            logger.warning('no environment variable set for %s', possible_names)
            return None
        raise ValueError(f'no environment variable set for {possible_names}')

    # Possible error: multiple values are not None and not equal
    if len(values) > 1 and len(set(values)) > 1:
        if not ignore_errors:
            raise ValueError(f'different environment variables set for {possible_names}: {values}')
        logger.warning('different environment variables set for %s: %s - using %s',
                       possible_names, values, values[0])

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
