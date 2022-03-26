# Rigid Body Transformation
# 
# Jin Cao <aihalop@gmail.com>
# 2019-12-2

import numpy as np
import unittest
import numbers
import operator

SMALL_NUMBER = 1e-10

normalize = lambda v: v / np.linalg.norm(v)

skew_symmetric = lambda v: np.array([[   0., -v[2],  v[1]],
                                     [ v[2],    0., -v[0]],
                                     [-v[1],  v[0],   0.]])


class Vector3(object):
    '''
    '''
    def __init__(self, x=0., y=0., z=0):
        self._x = x
        self._y = y
        self._z = z

    def x(self):
        return self._x

    def y(self):
        return self._y

    def z(self):
        return self._z

    def __iadd__(self, other):
        assert isinstance(other, Vector3)
        self._x += other.x()
        self._y += other.y()
        self._z += other.z()
        return self

    def __add__(self, other):
        if not isinstance(other, Vector3):
            raise ValueError("{} is not a type of Vector3".format(other))
        return Vector3(
            self._x + other.x(), self._y + other.y(), self._z + other.z()
        )

    def __radd__(self, other):
        if not isinstance(other, Vector3):
            raise ValueError("{} is not a type of Vector3".format(other))
        return Vector3(
            self._x + other.x(), self._y + other.y(), self._z + other.z()
        )

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            raise ValueError("{} is not a valid number.".format(other))
        return Vector3(other * self._x, other * self._y, other * self._z)

    def __rmul__(self, other):
        if not isinstance(other, numbers.Number):
            raise ValueError("{} is not a valid number.".format(other))
        return Vector3(other * self._x, other * self._y, other * self._z)

    def __eq__(self, other):
        if isinstance(other, Vector3):
            norm_difference = np.linalg.norm(
                (self._x - other.x(), self._y - other.y(), self._z - other.z())
            )
            return norm_difference < SMALL_NUMBER
        return False

    def normalized(self):
        return self * (1 / np.linalg.norm((self._x, self._y, self._z)))

    def __repr__(self):
        return "xyz: ({}, {}, {})".format(self._x, self._y, self._z)
    

class AxisAngle(object):
    def __init__(self, angle, axis):
        assert isinstance(axis, Vector3)
        self._angle = angle
        self._axis = axis

    def ToQuaternion(self):
        w = np.cos(self._angle * 0.5)
        v = np.sin(self._angle * 0.5) * self._axis.normalized()
        return Quaternion(w, v.x(), v.y(), v.z())
        

class Quaternion(object):
    '''
    '''
    def __init__(self, w=1., x=0., y=0., z=0.):
        self._scaler = w
        self._vector = Vector3(x, y, z)

    @staticmethod
    def identity():
        return Quaternion(1., 0., 0., 0.)

    def scalar(self):
        return self._scaler

    def vector(self):
        return np.array([self._vector.x(), self._vector.y(), self._vector.z()])

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            scalar = self.scalar() * other.scalar() \
                - np.dot(self.vector(), other.vector())
            vector = self.scalar() * other.vector() \
                + other.scalar() * self.vector() \
                + np.cross(self.vector(), other.vector())
            return Quaternion(scalar, *vector)
        elif isinstance(other, numbers.Number):
            return Quaternion(
                self.w() * other,
                self.x() * other,
                self.y() * other,
                self.z() * other
            )
        elif isinstance(other, Vector3):
            conjugation = self \
                * Quaternion(0, other.x(), other.y(), other.z()) \
                * self.conjugate()
            return Vector3(*conjugation.vector())
        else:
            raise ValueError(
                "Quaterion can not multiply a object of type {}"
                .format(type(other))
            )

    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Quaternion(
                self.w() * other,
                self.x() * other,
                self.y() * other,
                self.z() * other
            )
        else:
            raise ValueError(
                "An object of type {} multiply a quaternion is Not Defined."
                .format(type(other))
            )

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w() + other.w(),
                self.x() + other.x(),
                self.y() + other.y(),
                self.z() + other.z()
            )
        elif isinstance(other, numbers.Number):
            return self + Quaternion(other, 0, 0, 0)
        else:
            raise ValueError(
                "The operation of adding a value of type"
                "{} to a quaternion is Not Defined."
                .format(type(other))
            )
        

    def __eq__(self, other):
        if isinstance(other, Quaternion):
            norm_difference = np.linalg.norm(
                (self.w() - other.w(),
                 self.x() - other.x(),
                 self.y() - other.y(),
                 self.z() - other.z())
            )
            return norm_difference < SMALL_NUMBER
        return False

    def __repr__(self):
        return "(w: {:.16f}, x: {:.16f}, y: {:.16f}, z: {:.16f})".format(
            self.scalar(), *self.vector()
        )

    def inverse(self):
        return self.conjugate() * (1 / np.square(self.norm()))

    def matrix(self):
        v = self.vector()
        qv = np.reshape(v, (3, 1))
        R = (self.w() * self.w() - np.dot(v, v)) * np.identity(3) \
            + 2 * qv * qv.T + 2 * self.w() * skew_symmetric(self.vector())
        return R

    def conjugate(self):
        return Quaternion(self.w(), -self.x(), -self.y(), -self.z())

    def w(self):
        return self._scaler
    
    def x(self):
        return self._vector.x()

    def y(self):
        return self._vector.y()

    def z(self):
        return self._vector.z()

    def norm(self):
        return np.linalg.norm((self.w(), self.x(), self.y(), self.z()))
    
    def normalized(self):
        scale = 1. / self.norm()
        return Quaternion(
            self._w * scale,
            self._x * scale,
            self._y * scale,
            self._z * scale
        )

    def ToEuler(self):
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


class Translation(Vector3):
    pass

class Rotation(Quaternion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __mul__(self, another):
        if isinstance(another, Translation):
            x, y, z = super().__mul__(
                (another.x(), another.y(), another.z()))
            return Translation(x, y, z)
        else:
            return super().__mul__(another)

    def __inverse__(self):
        return Rotation(super().inverse())


class Rigid3D(object):
    def __init__(self, translation, rotation):
        self._translation = translation
        self._rotation = rotation

    def inverse(self):
        return Rigid3D(
            self._rotation.inverse() * self._translation,
            self._rotation.inverse()
        )

    def __mul__(self, another):
        return Rigid3D(
            self._rotation * another.translation() + self._translation,
            self._rotation * another.rotation()
        )

    def rotation(self):
        return self._rotation

    def translation(self):
        return self._translation

    def __repr__(self):
        message_template = \
            "tranlation(x,y,z), rotation(w,x,y,z):" \
            + "({}, {}, {}), ({}, {}, {}, {})"
        return message_template.format(
            self._translation.x(), self._translation.y(), self._translation.z(),
            self._rotation.w(), self._rotation.x(),
            self._rotation.y(), self._rotation.z()
        )


# class Rigid3D(Rigid):
#     pass


# class Rigid2D(Rigid):
#     def __init__(self):
#         pass

    

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


class TestVector3(unittest.TestCase):
    def test_vector_plus(self):
        v1 = Vector3(1., 1., 1.)
        v2 = Vector3(1., 2., 3.)
        v1 += v2
        self.assertEqual(v1 + v2, Vector3(3., 5., 7.))
        self.assertRaises(ValueError, operator.add, 1.0, v1)
        self.assertRaises(ValueError, operator.add, v1, 1.0)

    def test_vector_multiple(self):
        v = Vector3(1., 2., 3.)
        self.assertEqual(v * 2, Vector3(2., 4., 6.))
        self.assertEqual(3 * v, Vector3(3., 6., 9.))


class TestAngleAxis(unittest.TestCase):
    def test_angleaxis(self):
        axis_angle = AxisAngle(np.pi / 2, Vector3(0., 0., 2.))
        self.assertEqual(
            axis_angle.ToQuaternion(),
            Quaternion(0.7071067811865477, 0., 0., 0.7071067811865476))
        
'''
class TestRigid2D(unittest.TestCase):
    def setUp(self):
        self.A = Rigid2D(1, 2, np.pi/6)
        self.B = Rigid2D(3, 3, np.pi/4)
        self.C = Rigid2D(0, 0, np.pi / 4)
        print("A = {}, B = {}".format(self.A, self.B))

    def tearDown(self):
        pass

    def test_inverse(self):
        # print("A.inverse() * B = {}".format(self.A.inverse() * self.B))
        # print("C.inverse() * C = {}".format(self.C.inverse() * self.C))
        # print("C * C.inverse() = {}".format(self.C * self.C.inverse()))
        pass

    def test_vectorize(self):
        print("self.A.vectorize(): ", self.A.vectorize(), type(self.A.vectorize()))
        self.assertEqual(type(self.A.vectorize()), np.ndarray)
        np.testing.assert_array_almost_equal(
            self.A.vectorize(), [1.0, 2.0, 0.5235987]
        )

class TestRotation(unittest.TestCase):
    def setUp(self):
        self.rotation = Rotation(roll=0.0, pitch=0.0, yaw=0.575)

    def test_rotation(self):
        self.assertAlmostEqual(self.rotation.w(), 0.9589558)
        self.assertAlmostEqual(self.rotation.z(), 0.2835557)


class TestRigid3D(unittest.TestCase):
    def setUp(self):
        self.A = Rigid3D(Translation(0., 0., 0.),
                         Rotation(roll=0., pitch=0., yaw=np.pi / 2))
        self.B = Rigid3D(Translation(1., 0., 0.),
                         Rotation(1.0, 0., 0., 0.))
        self.C = Rigid3D(Translation(1., 0., 0.),
                         Rotation(roll=0., pitch=0., yaw=np.pi / 2))

    def test_multiplication(self):
        print("self.A: ", self.A)
        print(self.A.rotation() * self.A.translation())
        print(self.A * self.B)

    def test_inverse(self):
        print("self.C.inverse(): ", self.C.inverse())
'''

class TestQuaternion(unittest.TestCase):
    def setUp(self):
        self.q45 = AxisAngle(np.pi / 4, Vector3(0., 0., 1.)).ToQuaternion()
        self.q90 = AxisAngle(np.pi / 2, Vector3(0., 0., 1.)).ToQuaternion()
        self.v = Vector3(1., 0., 0.)
        self.q1234 = Quaternion(1, 2, 3, 4)

    def test_initialization(self):
        identity = Quaternion()
        self.assertTupleEqual(
            (identity.w(), identity.x(), identity.y(), identity.z()),
            (1.0, 0.0, 0.0, 0.0)
        )

    def test_multiple(self):
        self.assertEqual(self.q45 * self.q45, self.q90)
        self.assertEqual(self.q90 * self.v, Vector3(0., 1., 0.))
        self.assertEqual(2. * Quaternion(1., 2., 3., 4.),
                         Quaternion(2., 4., 6., 8.))
        self.assertRaises(ValueError, lambda: self.q45 * "invalid type")
        self.assertRaises(ValueError, lambda: [1., 0., 0.] * self.q90)
        self.assertRaises(ValueError, lambda: self.v * self.q90)

    def test_addition(self):
        q = Quaternion(1, 0, 0, 0) + Quaternion(0, 0, 1, 0)
        self.assertTupleEqual((q.w(), q.x(), q.y(), q.z()), (1, 0, 1, 0))
        q_and_scaler = Quaternion(0, 0, 0, 0) + 1
        self.assertTupleEqual(
            (q_and_scaler.w(), q_and_scaler.x(), q_and_scaler.y(), q_and_scaler.z()),
            (1, 0, 0, 0)
        )

        self.assertRaises(ValueError, lambda: q + "invalid type")

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
            np.reshape(self.q90.matrix(), 9) - np.array([0., -1., 0.,
                                                       1., 0., 0.,
                                                       0., 0., 1.]))
        self.assertAlmostEqual(diff_norm, 0.)


if __name__=="__main__":
    unittest.main(exit=False)
