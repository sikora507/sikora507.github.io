---
title: "PCA - Principal Component Analysis"
date: 2019-02-04
tags:
  - machine-learning
---

PCA is a popular method for dimensionality reduction.
It projects data from higher dimension on to hyperplane _(or in this case simple 2d plane)_ that is closest to the data, like 3d to 2d, 2d to 1d, but dimensions can be even higher.

This example shows projecting data from 3d space onto 2d plane.

```python
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
%matplotlib notebook
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

Create test 3D dataset:

```python
m=60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m)*3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)
```

Display test data

```python
fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], label='Test data')
ax.legend()
fig.show()
```

    <IPython.core.display.Javascript object>

<img src="Machine Learning/assets/img/pca-1.png" width="800">

## Using Singular Value Decomposition

Print first 20 rows of the data:

```python
print(X[0:20,:])
```

    [[ 0.7039474   0.65945649  0.3645693 ]
     [-1.0763742  -0.50036633 -0.21646923]
     [-0.90914129  0.22798983  0.05968884]
     [-0.26327638  0.52076062  0.31957985]
     [ 1.10617261  0.17601773  0.13888377]
     [ 1.0008109   0.15934054  0.07250963]
     [ 0.87852969 -0.25299137 -0.07699588]
     [-1.13697989 -0.29958862 -0.2851556 ]
     [-0.36237145  0.50948368  0.10889779]
     [-0.7732104   0.3332867   0.05678017]
     [ 0.77535051 -0.28414251  0.01996138]
     [-0.95258223 -0.54563776 -0.17623123]
     [-1.14143853 -0.19599228 -0.17164135]
     [ 1.10180912  0.27754754  0.33879858]
     [ 1.12822483  0.30165509  0.17685333]
     [ 1.16129536  0.28698597  0.47424224]
     [ 0.97281446  0.60223335  0.34051818]
     [ 0.05953046  0.59871039  0.09985041]
     [ 0.47966775  0.76970355  0.17178859]
     [ 0.9661238   0.46601271  0.28466344]]

Calculate mean for each column:

```python
X.mean(axis=0)
```

    array([0.15681044, 0.17698876, 0.0862971 ])

Make columns 0 mean (substract mean from the matrix):

```python
X_centered = X - X.mean(axis=0)
```

```python
X_centered[0:20,:]
```

    array([[ 5.47136958e-01,  4.82467738e-01,  2.78272202e-01],
           [-1.23318464e+00, -6.77355084e-01, -3.02766325e-01],
           [-1.06595173e+00,  5.10010710e-02, -2.66082639e-02],
           [-4.20086817e-01,  3.43771864e-01,  2.33282748e-01],
           [ 9.49362168e-01, -9.71028359e-04,  5.25866684e-02],
           [ 8.44000462e-01, -1.76482181e-02, -1.37874640e-02],
           [ 7.21719251e-01, -4.29980124e-01, -1.63292983e-01],
           [-1.29379033e+00, -4.76577375e-01, -3.71452702e-01],
           [-5.19181893e-01,  3.32494922e-01,  2.26006880e-02],
           [-9.30020839e-01,  1.56297940e-01, -2.95169326e-02],
           [ 6.18540077e-01, -4.61131262e-01, -6.63357193e-02],
           [-1.10939267e+00, -7.22626519e-01, -2.62528326e-01],
           [-1.29824897e+00, -3.72981033e-01, -2.57938446e-01],
           [ 9.44998681e-01,  1.00558780e-01,  2.52501481e-01],
           [ 9.71414397e-01,  1.24666339e-01,  9.05562297e-02],
           [ 1.00448492e+00,  1.09997216e-01,  3.87945146e-01],
           [ 8.16004024e-01,  4.25244590e-01,  2.54221086e-01],
           [-9.72799781e-02,  4.21721631e-01,  1.35533075e-02],
           [ 3.22857312e-01,  5.92714789e-01,  8.54914899e-02],
           [ 8.09313362e-01,  2.89023958e-01,  1.98366337e-01]])

Use np.linalg.svd() to compute singular value decomposition components: (U, s, Vt)

```python
U, s, Vt = np.linalg.svd(X_centered)
```

These 3 matrices can fully reconstruct our mean-centered 3d data by applying multiplication U * s * Vt. Vt holds all the **principal components** we are looking for.

```python
Vt
```

    array([[-0.95250178, -0.24902446, -0.17529172],
           [ 0.29267159, -0.9076305 , -0.30091563],
           [ 0.08416476,  0.33792558, -0.93740205]])

We are projecting from 3d to 2d so we need to take first 2 components from Vt matrix.

Extract first 2 principal components from Vt:

```python
c1 = Vt.T[:,0]
c2 = Vt.T[:,1]
```

```python
c1
```

    array([-0.95250178, -0.24902446, -0.17529172])

```python
c2
```

    array([ 0.29267159, -0.9076305 , -0.30091563])

They are needed to **create transformation matrix** that will map points from 3d space to 2d space

```python
np.column_stack((c1, c2))
```

    array([[-0.95250178,  0.29267159],
           [-0.24902446, -0.9076305 ],
           [-0.17529172, -0.30091563]])

```python
W2 = np.stack((c1,c2)).T
X2D = X_centered.dot(W2)
```

Plot the projection in 2D

```python
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(X2D[:,0], X2D[:,1], label='2d projection')
ax.legend()
ax.plot()
```

    <IPython.core.display.Javascript object>

<img src="Machine Learning/assets/img/pca-2.png" width="600">

    []

## PCA using Scikit-Learn

Same can be achieved with less amount of code using Scikit-Learn PCA:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
X2D_scikit = pca.fit_transform(X)
```

```python
fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(X2D_scikit[:,0], X2D_scikit[:,1], label='2d projection')
ax.legend()
ax.plot()
```

    <IPython.core.display.Javascript object>

<img src="Machine Learning/assets/img/pca-3.png" width="600">

    []

Comparison between SVD and ScikitLearn

```python
X2D[:5]
```

    array([[-0.690074  , -0.36150744],
           [ 1.39636097,  0.34497714],
           [ 1.00728461, -0.35025708],
           [ 0.2736333 , -0.50516373],
           [-0.91324535,  0.26290852]])

```python
X2D_scikit[:5]
```

    array([[-0.690074  , -0.36150744],
           [ 1.39636097,  0.34497714],
           [ 1.00728461, -0.35025708],
           [ 0.2736333 , -0.50516373],
           [-0.91324535,  0.26290852]])

- Scikit-Learn PCA does matrix mean centering automatically
- Sometimes one of the methods can result in flipped axes (multiplication by -1) but it's not so relevant for further learning on dataset with lower dimensions

### Inverse transform from 2d to 3d using Scikit-Learn

```python
X3D_inv = pca.inverse_transform(X2D_scikit)
```

```python
fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.scatter(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], label='Inversion using Scikit-Learn')
ax.legend()
fig.show()
```

    <IPython.core.display.Javascript object>

<img src="Machine Learning/assets/img/pca-4.png" width="800">

It's visible how both 3d charts are similar to each other, but not exactly the same. Even if recovered chart has 3 dimensions again and overall shape of the curve is similar, the points seem to be more flattened than on original chart.

There is some reconstruction error which can be computed:

```python
np.mean(np.sum(np.square(X3D_inv - X), axis=1))
```

    0.009421417196957218

### Inversion using numpy's SVD

```python
X3D_svd = X2D.dot(Vt[:2, :])
```

```python
fig = plt.figure(figsize=(8,6))
ax = fig.gca(projection='3d')
ax.scatter(X3D_svd[:, 0], X3D_svd[:, 1], X3D_svd[:, 2], label='Inversion with SVD')
ax.legend()
fig.show()
```

    <IPython.core.display.Javascript object>

<img src="Machine Learning/assets/img/pca-5.png" width="800">

The reconstruction error after reversing manually is:

```python
np.mean(np.sum(np.square(X3D_svd - X), axis=1))
```

    0.07278313972817166

**SVD error after reconstruction is higher than Scikit-Learn** It can be even observed on the 3d charts. Even if they look the same from the first glance, the axes are positioned slightly different. That's because Scikit-Learn automatically takes care of reversing the mean centering.

To achieve the same results as with Scikit-Learn's PCA we need to add mean to 3d reconstructed data manually:

```python
X3D_svd += X.mean(axis=0)
```

```python
np.mean(np.sum(np.square(X3D_svd - X), axis=1))
```

    0.009421417196957218

This experiment was made after reading Hands-On Machine Learning with Scikit-Learn and TensorFlow. If you still don't have it, I highly recommend buying it.

<a target="_blank"  href="https://www.amazon.com/gp/product/1491962291/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491962291&linkCode=as2&tag=sikora507-20&linkId=2afda09226851f789f24d5671b6d1ca3"><img border="0" src="https://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&MarketPlace=US&ASIN=1491962291&ServiceVersion=20070822&ID=AsinImage&WS=1&Format=_SL250_&tag=sikora507-20" ></a><img src="https://ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1491962291" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

<a target="_blank" href="https://www.amazon.com/gp/product/1491962291/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=1491962291&linkCode=as2&tag=sikora507-20&linkId=64bd5103ef0b10b13cba3936d277e786">Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems</a><img src="https://ir-na.amazon-adsystem.com/e/ir?t=sikora507-20&l=am2&o=1&a=1491962291" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />

Source code is available to download at my github repository:

<https://github.com/sikora507/pca-experiment>
