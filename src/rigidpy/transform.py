# Rigid Transfrom
# 
# Jin Cao <aihalop@gmail.com>
# 2019-12-2

import numpy as np
import unittest
import numbers

normalize = lambda v: v / np.linalg.norm(v)

skew_symmetric = lambda v: np.array([[   0., -v[2],  v[1]],
                                     [ v[2],    0., -v[0]],
                                     [-v[1],  v[0],   0.]])


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
        else:
            self._w = 1.0;
            self._x = 0.0;
            self._y = 0.0;
            self._z = 0.0;
            print("len(args): {}".format(len(args)))

    @staticmethod
    def identity():
        return Quaternion(1., 0., 0., 0.)

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

    def inverse(self):
        return self.conjugate() * (1 / np.square(self.norm()))

    def matrix(self):
        v = self.vector()
        qv = np.reshape(v, (3, 1))
        R = (self._w * self._w - np.dot(v, v)) * np.identity(3) \
            + 2 * qv * qv.T + 2 * self._w * skew_symmetric(self.vector())
        return R

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

    def norm(self):
        return np.sqrt(self._w * self._w +
                       self._x * self._x +
                       self._y * self._y +
                       self._z * self._z)
    
    def normalized(self):
        n = self.norm()
        return Quaternion(self._w / n, self._x / n, self._y / n, self._z / n)


    def to_Euler(self):
        pass


def euler_to_quaterion(roll, pitch, yaw):
    return Quaternion(angle=roll, axis=(1., 0., 0.)) \
        * Quaternion(angle=pitch, axis=(0., 1., 0.)) \
        * Quaternion(angle=yaw, axis=(0., 0., 1.))


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
    def __init__(self, x=0., y=0., z=0., roll=0., pitch=0., yaw=0.):
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
        self.euler_rpy = (0.2, 0.4, 0.6)

        self.q = Quaternion(angle=np.pi / 2, axis=(0.0, 0.0, 1.0))
        self.q1234 = Quaternion(1, 2, 3, 4)

    def treaDown(self):
        pass

    def test_initialization(self):
        identity = Quaternion()
        self.assertTupleEqual(
            (identity.w(), identity.x(), identity.y(), identity.z()),
            (1.0, 0.0, 0.0, 0.0)
        )

    def test_multiple(self):
        q2 = self.q * self.q
        diff = np.array([q2.w(), q2.x(), q2.y(), q2.z()]) - np.array([0., 0., 0., 1.0])
        self.assertAlmostEqual(np.linalg.norm(diff), 0.0)

    def test_addition(self):
        q = Quaternion(1, 0, 0, 0) + Quaternion(0, 0, 1, 0)
        self.assertTupleEqual((q.w(), q.x(), q.y(), q.z()), (1, 0, 1, 0))
        q_and_scaler = Quaternion(0, 0, 0, 0) + 1
        self.assertTupleEqual(
            (q_and_scaler.w(), q_and_scaler.x(), q_and_scaler.y(), q_and_scaler.z()),
            (1, 0, 0, 0)
        )

    def test_quaternion_from_two_vectors(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        q = quaternion_from_two_vectors(v1, v2)
        diff = np.array([q.w(), q.x(), q.y(), q.z()]) - np.array([0.707107, 0, 0, 0.707107])
        self.assertAlmostEqual(np.linalg.norm(diff), 0.0, 6)
        
    def test_conjugate(self):
        qc = self.q1234.conjugate()
        self.assertTupleEqual((qc.w(), qc.x(), qc.y(), qc.z()), (1, -2, -3, -4))

    def test_norm(self):
        self.assertAlmostEqual(self.q1234.norm(), 5.4772, 4)

    def test_inverse(self):
        q = self.q1234.inverse() * self.q1234
        i = Quaternion.identity()
        self.assertTupleEqual(
            (q.w(), q.x(), q.y(), q.z()),
            (i.w(), i.x(), i.y(), i.z())
        )

    def test_matrix(self):
        diff_norm = np.linalg.norm(
            np.reshape(self.q.matrix(), 9) - np.array([0., -1., 0.,
                                                       1., 0., 0.,
                                                       0., 0., 1.]))
        self.assertAlmostEqual(diff_norm, 0.)


if __name__=="__main__":
    unittest.main()
