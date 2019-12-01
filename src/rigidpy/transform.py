import numpy as np
import unittest

class Quaternion(object):
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self._w = w
        self._x = x
        self._y = y
        self._z = z

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
            self.translation(), np.arccos(self._rotation[0, 0])
        )


class TestRigid2D(unittest.TestCase):
    def setUp(self):
        self.A = Rigid2D(1, 2, np.pi/6)
        self.B = Rigid2D(3, 3, np.pi/4)
        print(self.A)

    def tearDown(self):
        pass

    def test_inverse(self):
        print(self.A.inverse() * self.B)


class TestQuaternion(unittest.TestCase):
    def setUp(self):
        self.Q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.Q2 = Quaternion(2.0, 3.0, 4.0, 5.0)

    def test_initialization(self):
        print("Q: {}".format(self.Q))

    def test_multiple(self):
        print("Q: {} * Q2: {} = \n{}".format(self.Q, self.Q2, self.Q * self.Q2))


if __name__=="__main__":
    unittest.main()
