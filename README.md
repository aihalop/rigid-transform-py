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
```

### Transform 3D Vector

```python
from rigid_transform import Rigid3, Translation, Rotation, Vector3
import math

T = Rigid3(
    Translation(x=1., y=0., z=0.), 
    Rotation(roll=0., pitch=0., yaw=math.pi / 2)
)
v = Vector3(x=1., y=0., z=0.)
print(T * v) # Vector3(xyz: (1.0000, 1.0000, 0.0000))
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
```
