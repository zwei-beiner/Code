from abc import ABC, abstractmethod

class AbstractConstraint(ABC):
    def __init__(self, type):
        self._type = type

    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self, value):
        pass


class FixedConstraint(AbstractConstraint):
    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not (type(value) is self._type):
            raise TypeError(f'Incorrect type: {value}')
        self._value = value


class BoundedConstraint(AbstractConstraint):
    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not (type(value) is tuple and list(map(type, value)) == [self._type, self._type]):
            raise TypeError(f'Incorrect type: {value}')
        self._value = value


class CategoricalConstraint(AbstractConstraint):
    def __init__(self, type, value):
        super().__init__(type)
        self.value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if not (type(value) is list and all(type(v) is self._type for v in value)):
            TypeError(f'Incorrect type: {value}')
        self._value = value