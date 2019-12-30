# Rigid Transformation objects.
# 
# Jin Cao <aihalop@gmail.com>
# 2019-12-2

import numpy as np
import unittest

class Quaternion(object):
    def __init__(self, *args, **kwargs):
        if args and not kwargs:
            w, x, y, z = args
            self._w = w
            self._x = x
            self._y = y
            self._z = z
        elif not args and kwargs:
            pass

    def scalar(self):
        return self._w

    def vector(self):
        return np.array([self._x, self._y, self._z])

    def __mul__(self, Q):
        scalar = self.scalar() * Q.scalar() - \
                 np.dot(self.vector(), Q.vector())
        vector = self.scalar() * Q.vector() + Q.scalar() * self.vector() + \
                 np.cross(self.vector(), Q.vector())
        return Quaternion(scalar, vector[0], vector[1], vector[2])

    def __str__(self):
        return "(w: {}, x: {}, y: {}, z:{})".format(
            self.scalar(), *self.vector()
        )

    def matrix(self):
        pass

    def to_list(self):
        return np.array([self._w, self._x, self._y, self._z])


class Rigid2D(object):
    def __init__(self, x=0, y=0, theta=0):
        self._rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        self._translation = np.array([x, y])

    def rotation(self):
        return self._rotation

    def translation(self):
        return self._translation

    def inverse(self):
        T = Rigid2D()
        T._rotation = self._rotation.T
        T._translation = -1 * np.dot(T._rotation, self._translation)
        return T

    def homogenity(self):
        return np.concatenate((
            np.concatenate((
                self._rotation,
                np.expand_dims(self._translation, axis=0).T), axis=1
            ),
            [[0, 0, 1]]), axis=0
        )

    def __mul__(self, B):
        C = Rigid2D()
        C._rotation = np.dot(self.rotation(), B.rotation())
        C._translation = np.dot(self.rotation(), B.translation()) + \
                         self.translation()
        return C

    def __str__(self):
        return "({}, {})".format(
            self.translation(),
            self.angle()
        )

    def angle(self):
        return np.arctan2(self._rotation[1, 0], self._rotation[0, 0])

    def vectorize(self):
        return np.concatenate([self.translation(), np.array([self.angle()])])


class TestRigid2D(unittest.TestCase):
    def setUp(self):
        self.A = Rigid2D(1, 2, np.pi/6)
        self.B = Rigid2D(3, 3, np.pi/4)
        self.C = Rigid2D(0, 0, np.pi / 4)
        print("A = {}, B = {}".format(self.A, self.B))

    def tearDown(self):
        pass

    def test_inverse(self):
        print("A.inverse() * B = {}".format(self.A.inverse() * self.B))
        print("C.inverse() * C = {}".format(self.C.inverse() * self.C))
        print("C * C.inverse() = {}".format(self.C * self.C.inverse()))

    def test_vectorize(self):
        print("self.A.vectorize(): ", self.A.vectorize(), type(self.A.vectorize()))
        self.assertEqual(type(self.A.vectorize()), np.ndarray)
        np.testing.assert_array_almost_equal(
            self.A.vectorize(), [1.0, 2.0, 0.5235987]
        )


class TestQuaternion(unittest.TestCase):
    def setUp(self):
        # angle = pi/6, axis = (1., 0., 0.)
        self.Q = Quaternion(0.9659258262890683, 0.25881904510252074, 0.0, 0.0)

    def test_initialization(self):
        print("Q: {}".format(self.Q))

    def test_multiple(self):
        q1 = self.Q * self.Q
        q2 = Quaternion(0.8660254037844387, 0.49999999999999994, 0.0, 0.0)
        np.testing.assert_array_almost_equal(q1.to_list(), q2.to_list())


if __name__=="__main__":
    unittest.main()
