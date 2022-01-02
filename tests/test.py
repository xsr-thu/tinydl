import numpy as np
from unittest import TestCase
import unittest
import os
import sys

target_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "build")

print(target_dir)
sys.path.insert(0, target_dir)

import tinydl


class TestTensor(TestCase):
    def test_1d(self):
        data = np.random.randn(3).astype(np.float32)
        t = tinydl.Tensor(data)
        r = t.to_numpy()
        self.assertEqual(data.shape, r.shape)
        self.assertTrue((data == r).all())

    def test_2d(self):
        data = np.random.randn(128, 128).astype(np.float32)
        t = tinydl.Tensor(data)
        r = t.to_numpy()
        self.assertEqual(data.shape, r.shape)
        self.assertTrue((data == r).all())


    def test_binary_op(self):
        x = np.random.randn(128).astype(np.float32)
        y = np.random.randn(128).astype(np.float32)
        tx = tinydl.Tensor(x)
        ty = tinydl.Tensor(y)
        
        z = x + y
        tz = tinydl.op_add(tx, ty)
        tz_np = tz.to_numpy()
        self.assertTrue((z == tz_np).all())
        
        z = x - y
        tz = tinydl.op_sub(tx, ty)
        tz_np = tz.to_numpy()
        self.assertTrue((z == tz_np).all())

        z = x * y
        tz = tinydl.op_mul(tx, ty)
        tz_np = tz.to_numpy()
        self.assertTrue((z == tz_np).all())

        z = x / y
        tz = tinydl.op_div(tx, ty)
        tz_np = tz.to_numpy()
        self.assertTrue((z == tz_np).all())

    def test_matmul(self):
        x = np.eye(33).astype("float32")
        y = np.ones((33, 32)).astype("float32")
        x = tinydl.Tensor(x)
        y = tinydl.Tensor(y)
        z = tinydl.op_matmul(x, y)
        z = z.to_numpy()
        print(z)
        print(z.shape)


if __name__ == "__main__":
    unittest.main()

