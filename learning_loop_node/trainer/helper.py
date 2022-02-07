from uuid import UUID


def is_valid_uuid4(val):
    try:
        UUID(str(val)).version
        return True
    except ValueError:
        return False
