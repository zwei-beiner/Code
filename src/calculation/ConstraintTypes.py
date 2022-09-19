from abc import ABC, abstractmethod


class AbstractConstraint(ABC):
    """
    Abstract class for all constraint types.

    Motivation:
        The specific constraints which have to be implemented are
        (1) A variable takes a fixed value (FixedConstraint), e.g.
            var = 5
        (2) A variable can take a value from a discrete set of values (CategoricalConstraint), e.g.
            var = 0 or var = 1 or var = 2
        (3) A variable can take a value in a range of values (BoundedConstraint), e.h.
            var takes some value in the interval 3.2 to 6.7
        All of these constraints store a number of values. For the above examples, these are
        (1) the integer 5
        (2) the integers 0, 1 and 2
        (3) the floats 3.2 and 6.7

        This abstract base class stores the type of these values. It provides basic infrastructure which is common
        to all three constraint types.
        The type of the value is stored because this allows to check the type when a new AbstractConstraint object is
        created. This is important as user input must be parsed, which makes type checks necessary.
    """

    def __init__(self, type):
        """
        @param type: Type of the value of the constraint
        """
        self._type = type

    @property
    @abstractmethod
    def value(self):
        """
        Abstract method which serves as a getter for self.value.
        """
        pass

    @value.setter
    @abstractmethod
    def value(self, value):
        """
        Abstract method which serves as a setter for self.value.
        """
        pass


class FixedConstraint(AbstractConstraint):
    """
    Implements a fixed constraint, i.e. a variable takes a fixed value.
    """

    def __init__(self, type, value):
        # Store the type of the value in the AbstractConstraint parent class.
        super().__init__(type)
        # Store the value itself.
        self.value = value

    @property
    def value(self):
        """
        Getter method for self._value.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Setter method for self._value.
        """
        # Check if the type is correct.
        if not (type(value) is self._type):
            raise TypeError(f'Incorrect type: {value}')
        # Store value.
        self._value = value


class BoundedConstraint(AbstractConstraint):
    """
    Implements a constraint in which a variable is bounded, i.e. a variable takes values in a range specified by two
    bounds.
    """

    def __init__(self, type, value):
        # Store the type of the value in the AbstractConstraint parent class.
        super().__init__(type)
        # Store the value itself.
        self.value = value

    @property
    def value(self):
        """
        Getter method for self._value.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Setter method for self._value.
        """
        # Check if 'value' is a tuple and each entry of the tuple has the correct type self._type.
        if not (type(value) is tuple and list(map(type, value)) == [self._type, self._type]):
            raise TypeError(f'Incorrect type: {value}')
        # Store value.
        self._value = value


class CategoricalConstraint(AbstractConstraint):
    """
    Implements a constraint in which a variable can take a number of values, i.e. it takes values from a specified list
    of values.
    """

    def __init__(self, type, value):
        # Store the type of the value in the AbstractConstraint parent class.
        super().__init__(type)
        # Store the value itself.
        self.value = value

    @property
    def value(self):
        """
        Getter method for self._value.
        """
        return self._value

    @value.setter
    def value(self, value):
        """
        Setter method for self._value.
        """
        # Check if value is a list and each entry in the list has type self._type.
        if not (type(value) is list and all(type(v) is self._type for v in value)):
            TypeError(f'Incorrect type: {value}')
        # Store value.
        self._value = value
