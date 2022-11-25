

class Errors():
    def __init__(self):
        self._errors: dict = {}

    def set(self, key: str, value: str):
        self._errors[key] = value

    def reset(self, key: str):
        try:
            del self._errors[key]
        except AttributeError:
            pass
        except KeyError:
            pass

    def reset_all(self):
        self._errors = {}

    def has_error_for(self, key: str) -> bool:
        return key in self._errors

    def has_error(self) -> bool:
        return self._errors == {}


class TrainingError(Exception):
    def __init__(self, cause: str, *args: object) -> None:
        super().__init__(*args)
        self.cause = cause
