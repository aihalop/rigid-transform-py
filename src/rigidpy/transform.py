import numpy as np
import unittest


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


if __name__=="__main__":
    unittest.main()
