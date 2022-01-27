# Rigid Transformation objects.
# 
# Jin Cao <aihalop@gmail.com>
# 2019-12-2

import numpy as np
import unittest
import numbers

normalize = lambda v: v / np.linalg.norm(v)

class Quaternion(object):
    '''
    '''
    def __init__(self, *args, **kwargs):
        if args and not kwargs:
            w, x, y, z = args
            self._w = w
            self._x = x
            self._y = y
            self._z = z
        elif not args and kwargs:
            axis = kwargs.get("axis")
            angle = kwargs.get("angle")
            if (axis is not None) and (angle is not None):
                self._w = np.cos(angle * 0.5)
                self._x, self._y, self._z = np.sin(angle * 0.5) * normalize(axis)
            else:
                print("Wrong argument.")
                

    def scalar(self):
        return self._w

    def vector(self):
        return np.array([self._x, self._y, self._z])

    def __mul__(self, q):
        if isinstance(q, Quaternion):
            scalar = self.scalar() * q.scalar() - \
                     np.dot(self.vector(), q.vector())
            vector = self.scalar() * q.vector() + q.scalar() * self.vector() + \
                     np.cross(self.vector(), q.vector())
            return Quaternion(scalar, *vector)
        elif isinstance(q, numbers.Number):
            return Quaternion(
                self._w * q, self._x * q, self._y * q, self._z * q
            )
        elif type(q) in (list, tuple, np.ndarray):
            assert(len(q) == 3)
            return self * Quaternion(0, *q)


    def __add__(self, q):
        if isinstance(q, Quaternion):
            return Quaternion(
                self._w + q.w(),
                self._x + q.x(), self._y + q.y(), self._z + q.z()
            )
        elif isinstance(q, numbers.Number):
            return self + Quaternion(q, 0, 0, 0)

    def __repr__(self):
        return "(w: {:.3f}, x: {:.3f}, y: {:.3f}, z: {:.3f})".format(
            self.scalar(), *self.vector()
        )

    def matrix(self):
        pass

    def to_list(self):
        return np.array([self._w, self._x, self._y, self._z])

    def conjugate(self):
        return Quaternion(self._w, -self._x, -self._y, -self._z)

    def w(self):
        return self._w
    
    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def normalize(self):
        norm = np.linalg.norm(self.to_list())
        return Quaternion(*(self.to_list() / norm))

    def to_Euler(self):
        pass


def euler_to_quaterion(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5);
    sy = np.sin(yaw * 0.5);
    cp = np.cos(pitch * 0.5);
    sp = np.sin(pitch * 0.5);
    cr = np.cos(roll * 0.5);
    sr = np.sin(roll * 0.5);
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return Quaternion(w, x, y, z)


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


class Rigid3D(object):
    def __init__(self, x, y, z, roll, pitch, yaw):
        self.translation = np.array([x, y, z])
        self.quaternion = euler_to_quaterion(roll, pitch, yaw)

    def inverse(self):
        pass

    def __mul__(self, B):
        pass



def quaternion_from_two_vectors(v1, v2):
    assert(len(v1) == 3)
    assert(len(v2) == 3)
    v1 = normalize(v1)
    v2 = normalize(v2)
    rotation_vector = np.cross(v1, v2)
    cos_angle = np.dot(v1, v2)
    q0 = np.sqrt((cos_angle + 1) * 0.5)
    qn = np.sqrt((1 - cos_angle) * 0.5) * rotation_vector
    return Quaternion(q0, qn[0], qn[1], qn[2])
    

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
        self.Q90y = Quaternion(angle=-np.pi / 2, axis=(0, 1, 0))

    def treaDown(self):
        pass
        
    def test_initialization(self):
        print("Q: {}".format(self.Q))

    def test_multiple(self):
        q1 = self.Q * self.Q
        q2 = Quaternion(0.8660254037844387, 0.49999999999999994, 0.0, 0.0)
        np.testing.assert_array_almost_equal(q1.to_list(), q2.to_list())

        print("Quaternion(1, 0, 0, 0) * 0.2 = {}".format(Quaternion(1, 0, 0, 0) * 0.2))
        print("q * v = {}".format(self.Q * [1, 0, 0]))

    def test_addition(self):
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0, 1, 0, 0)
        print("q1 + q2 = {}".format(q1 + q2))
        print("q1 + 1 = {}".format(q1 + 1))

    def test_rotation(self):
        print("Q90y = {}".format(self.Q90y))
        v = (1, 0, 0)
        print("q * v * qc = {}".format(self.Q90y * v * self.Q90y.conjugate()))

    def test_quaternion_from_two_vectors(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        q = quaternion_from_two_vectors(v1, v2)
        print("test_quaternion_from_two_vectors", q)
        np.testing.assert_array_almost_equal(
            q.to_list(), Quaternion(0.707107, 0, 0, 0.707107).to_list()
        )
        
    def test_normalize(self):
        q = Quaternion(0, 1, 2, 3)
        print("{} normalize to be {}".format(q, q.normalize()))


if __name__=="__main__":
    unittest.main()
