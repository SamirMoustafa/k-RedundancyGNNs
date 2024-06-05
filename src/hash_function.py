import hashlib
from pickle import PicklingError, dumps
from threading import Lock

from torch import is_tensor


class HashedValue(str):
    """
    A class that inherits from str and can be used as a hash value.
    """

    pass


class InvertibleHash:
    """
    A class that can be used to hash objects and invert the hash.
    """

    _instance = None
    _lock = Lock()

    def __init__(self):
        """
        Initializes the hash function.
        """
        # Acquire the lock while initializing
        with self._lock:
            self.hash_to_obj = getattr(self, "hash_to_obj", {})

    def __new__(cls, *args, **kwargs):
        """
        Makes sure that only one instance of the hash function exists (Singleton Design Pattern).
        """
        # Acquire the lock while creating the instance
        with cls._lock:
            if not isinstance(cls._instance, cls):
                cls._instance = super(InvertibleHash, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def hash(self, obj):
        """
        Hashes an object and stores the mapping in both directions.
        """
        # Avoid double hashing
        if isinstance(obj, HashedValue):
            return obj
        # Converts PyTorch Tensors into a byte array
        if is_tensor(obj):
            obj = obj.cpu().numpy()
        try:
            # Try to pickle the object
            b = dumps(obj)
        except PicklingError:
            # If the object can't be pickled, convert it into a string
            b = str(obj).encode()

        # Create hash from bytes
        hash_val = HashedValue(hashlib.sha512(b).hexdigest())

        # Acquire the lock while modifying hash_to_obj
        with self._lock:
            # Store the mapping in both directions
            self.hash_to_obj[hash_val] = obj
        return hash_val

    def invert(self, hash_val):
        """
        Inverts a hash value and returns the original object.
        """
        invertible_hash = self.hash_to_obj[hash_val]
        while isinstance(invertible_hash, HashedValue):
            invertible_hash = self.hash_to_obj[invertible_hash]
        return invertible_hash


invertible_hash = InvertibleHash()


def hash(string):
    """
    A perfect hash function, that computes a shorter string (or integer maybe?) from parameter string

    :param string: String to be hashed
    :returns: a unique short representation of string
    """
    return invertible_hash.hash(string)


def inverse_hash(hashed_string):
    """
    Inverse function to hash

    :param hashed_string: String to be inverted
    :returns: original tensor
    """
    return invertible_hash.invert(hashed_string)
