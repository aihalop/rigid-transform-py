# Rigid Transform Py

This library provides classes to manipulate rigid transformation.

![rigid transform illustration](images/transform.svg "transform illustration")

## Install

```Shell
pip install rigid-transform-py
```

## Usage

### Transform 3D Rigid Body

```python
from rigid_transform import Rigid3, Translation, Rotation
import math

T12 = Rigid3(
    Translation(x=1., y=0., z=0.), 
    Rotation(roll=0., pitch=0., yaw=math.pi / 2)
)
T23 = Rigid3(
    Translation(x=1., y=0., z=0.), 
    Rotation(roll=0., pitch=0., yaw=0.)
)
T13 = T12 * T23
print(T13) # Rigid3(T xyz: (1.0000, 1.0000, 0.0000), R wxyz: (0.7071, 0.0000, 0.0000, 0.7071))

assert(T23 == T12.inverse() * T13)
assert(T12 == T13 * T23.inverse())

# access the elements of the Rigid3 object.
print("x: {}, y: {}, z: {}".format(
	T13.translation.x, T13.translation.y, T13.translation.z)
) # x: 1.0, y: 1.0000000000000002, z: 0.0

print("w: {}, x: {}, y: {}, z: {}".format(
	T13.rotation.w, T13.rotation.x, T13.rotation.y, T13.rotation.z)
) # w: 0.7071067811865477, x: 0.0, y: 0.0, z: 0.7071067811865476

roll, pitch, yaw = T13.rotation.ToEuler()
print("roll: {}, pitch: {}, yaw: {}".format(roll, pitch, yaw)) # roll: 0.0, pitch: 0.0, yaw: 1.5707963267948968
```

### Transform 3D Vector

```python
from rigid_transform import Rigid3, Translation, Rotation, Vector3
import math

T = Rigid3(
    Translation(x=1., y=0., z=0.), 
    Rotation(roll=0., pitch=0., yaw=math.pi / 2)
)
v1 = Vector3(x=1., y=0., z=0.)
v2 = T * v1
print(v2) # Vector3(xyz: (1.0000, 1.0000, 0.0000))

# access the elements of the Vector3 object
print("x: {}, y: {}, z: {}".format(v2.x, v2.y, v2.z)) # x: 1.0, y: 1.0000000000000002, z: 0.0
```

### Transform 2D Rigid Body

```python
from rigid_transform import Rigid2, Translation, Rotation
import math

T12 = Rigid2(x=1., y=0., theta=math.pi / 2)
T23 = Rigid2(x=1., y=0., theta=0.)
T13 = T12 * T23
print(T13) # Rigid2(x,y,theta: 1.0000, 1.0000, 1.5708)

assert(T23 == T12.inverse() * T13)
assert(T12 == T13 * T23.inverse())

# access the elements of the Rigid2 object
print("x: {}, y: {}, theta: {}".format(T13.x, T13.y, T13.theta)) # x: 1.0, y: 1.0000000000000002, theta: 1.570796326794897
```
