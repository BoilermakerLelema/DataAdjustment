1. 3DCoordinateTransformation.py:

The LS model used is indirect least squre: v = A*delta - f

The code is a simple realization of coordinate transformation by using elemental rotation.
Totally, there are seven unknowns: scale (1), rotaion angles (3), and translation (3).

2. GeneralLeastSquare.py

The LS model used is General Least Square: A*v + B*delta + f = 0

In this code, the method is applied to estimate the four paramters of a sphere.
