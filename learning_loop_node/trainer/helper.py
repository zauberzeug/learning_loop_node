from uuid import UUID


def is_valid_uuid4(val):
    try:
        return UUID(str(val)).version == 4.0
    except ValueError:
        return False
