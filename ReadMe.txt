1. 3DCoordinateTransformation.py:

The LS model used is indirect least squre: v = A*delta - f

The code is a simple realization of coordinate transformation by using elemental rotation.
Totally, there are seven unknowns: scale (1), rotaion angles (3), and translation (3).

2. GeneralLeastSquare.py

The LS model used is General Least Square: A*v + B*delta + f = 0

In this code, the method is applied to estimate the four paramters of a sphere.

The data used is in file "data_Sphere.txt".

3. PosteriorErrorAnalysis.py

In this example, we do the posterior error analysis on a network. We calculate the posterior sigma, variance, hypothesis test, confidence ellipse. 

'data_1a.txt': angle observations. 
'data_1c.txt': coordinates of known points.
'data_1d.txt': distance observations.

The network should be like the picture "error_ellipse.png". 

Attached also includes a matlab code to draw error ellipse. 
