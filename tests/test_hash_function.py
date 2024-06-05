from pickle import PicklingError
from unittest import TestCase, main

import numpy as np
from torch import tensor

from src.hash_function import InvertibleHash


class TestInvertibleHash(TestCase):
    def test_singleton(self):
        # Testing Singleton design pattern
        instance1 = InvertibleHash()
        instance2 = InvertibleHash()
        self.assertEqual(id(instance1), id(instance2))

    def test_hashing_and_inverting_simple_objects(self):
        # Testing basic objects
        objects = [42, "hello", [1, 2, 3], {"a": 1, "b": 2}]

        for obj in objects:
            hash_val = InvertibleHash().hash(obj)
            inverted_obj = InvertibleHash().invert(hash_val)
            self.assertEqual(obj, inverted_obj)

    def test_hashing_and_inverting_tensors(self):
        # Testing PyTorch tensors
        t = tensor([1.0, 2.0, 3.0])
        hash_val = InvertibleHash().hash(t)
        inverted_obj = InvertibleHash().invert(hash_val)
        np.testing.assert_array_equal(t.cpu().numpy(), inverted_obj)

    def test_hashing_and_inverting_unpicklable_objects(self):
        # Testing objects that can't be pickled
        class Unpicklable:
            def __reduce__(self):
                raise PicklingError

        obj = Unpicklable()
        hash_val = InvertibleHash().hash(obj)
        inverted_obj = InvertibleHash().invert(hash_val)
        self.assertEqual(str(obj), str(inverted_obj))

    def test_invert_unknown_hash(self):
        # Testing inversion of an unknown hash
        with self.assertRaises(KeyError):
            InvertibleHash().invert("unknown_hash")


if __name__ == "__main__":
    main()
