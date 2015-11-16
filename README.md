
Import the torch package under the name th (★☆☆☆☆)


```lua
th = require 'torch'
```

Print the torch version and the configuration (★☆☆☆☆)


```lua

```

Create a null vector of size 10 (★☆☆☆☆)


```lua
Z = th.zeros(10)
print(Z)
```

How to get the documentation of the torch add function from the command line ? (★☆☆☆☆)


```lua

```

9.Create a null vector of size 10 but the fifth value which is 1 (★☆☆☆☆)


```lua
Z = th.zeros(10)
Z[5] = 1 -- note that lua has 1-indexing
print(Z)
```

10.Create a vector with values ranging from 10 to 49 (★☆☆☆☆)


```lua
Z = th.range(10,49)
print(Z)
```

Reverse a vector (first element becomes last) (★☆☆☆☆)


```lua

```

Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆☆☆)


```lua
Z = th.range(0,8)
Z = Z:reshape(3,3)
print(Z)
```


```lua
Z = torch.Tensor(3,3)
i = 0
Z:apply(function() i = i + 1; return i end)
print(Z)
```

Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆☆☆)


```lua
Z = th.Tensor{1,2,0,0,4,0}
print(Z:nonzero())
```

Create a 3x3 identity matrix (★☆☆☆☆)


```lua
Z = th.eye(3)
print(Z)
```

Create a 3x3x3 array with random values (★☆☆☆☆)


```lua
Z = th.rand(3,3,3)
print(Z)
```

Create a 10x10 array with random values and find the minimum and maximum values (★☆☆☆☆)


```lua
Z = th.rand(10,10)
Zmin = Z:min()
print(Zmin)
Zmax = Z:max()
print(Zmax)
```

Create a random vector of size 30 and find the mean value (★☆☆☆☆)


```lua
Z = th.rand(30)
Zmean = Z:mean()
print(Zmean)
```

Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★★☆☆☆)


```lua

```

Create a 8x8 matrix and fill it with a checkerboard pattern (★★☆☆☆)


```lua
Z = th.zeros(8,8)
```

Create a checkerboard 8x8 matrix using the repeat function (★★☆☆☆)


```lua
Z = th.Tensor{{0,1},{1,0}}
print(Z:repeatTensor(4,4))
```

Normalize a 5x5 random matrix (★★☆☆☆)


```lua
Z = th.rand(5,5)
Zmax, Zmin = Z:max(), Z:min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)
```

Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★★☆☆☆)


```lua
A = th.ones(5,3)
B = th.ones(3,2)
print(A*B)
```

Create a 5x5 matrix with row values ranging from 0 to 4


```lua
A = th.range(0,4)
A = A:reshape(1,5)
print(A:expand(5,5))
```

Consider two random array A anb B, check if they are equal (★★☆☆☆)


```lua
A = th.rand(2,2)
B = th.rand(2,2)
print(th.all(A:eq(B)))
```

Make an array immutable (read-only) (★★☆☆☆)


```lua

```

Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆☆☆)


```lua
Z = th.rand(10,2)
X = Z[{{},1}]
Y = Z[{{},2}]
R = th.sqrt(X:pow(2)+Y:pow(2))
T = th.atan2(Y,X)
print(th.cat(R,T,2))
```

Create random vector of size 10 and replace the maximum value by 0 (★★☆☆☆)


```lua
Z = th.rand(10)
max,idx = Z:max(1)
Z[idx] = 0
print(Z)
```

Create a structured array with x and y coordinates covering the [0,1]x[0,1] area (★★☆☆☆)


```lua

```

Print the minimum and maximum representable value for each numpy scalar type (★★☆☆☆)


```lua

```

Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆☆☆)


```lua

```

Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆☆☆)


```lua
XY = th.rand(10,2)
Xi = XY[{{},1}]:reshape(1,10):expand(10,10)
Xij = (Xi-Xi:t()):pow(2)
Yi = XY[{{},2}]:reshape(1,10):expand(10,10)
Yij = (Yi-Yi:t()):pow(2)
dist = (Xij+Yij):sqrt()
print(dist)
```

Consider the following file: How to read it ? (★★☆☆☆)
1,2,3,4,5
6,,,7,8
,,9,10,11
Generate a generic 2D Gaussian-like array (★★☆☆☆)


```lua
X = th.linspace(-1,1,10):reshape(1,10):expand(10,10)
print(X:pow(2))
Y = th.linspace(-1,1,10):reshape(10,1):expand(10,10)
print(Y)
D = th.sqrt(X:pow(2)+Y:pow(2))
print(D)
```

How to randomly place p elements in a 2D array ? (★★★☆☆)


```lua

```

Subtract the mean of each row of a matrix (★★★☆☆)


```lua
X = th.rand(5,10)
M = X:mean(2):reshape(5,1):expand(X:size())
X = X - M
print(X)
```

How to I sort an array by the nth column ? (★★★☆☆)


```lua
Z = th.rand(5,10)
max,idx = Z[{{},3}]:sort(1)
print(Z:index(1,idx))
```

How to tell if a given 2D array has null columns ? (★★★☆☆)


```lua

```

Find the nearest value from a given value in an array (★★★☆☆)


```lua
Z = th.rand(10)
z = 0.5
min, idx = th.min(th.abs(Z-z),1)
print(idx)
print(Z[idx])
```

Consider a generator function that generates 10 integers and use it to build an array (★★★☆☆)


```lua

```

Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices) ? (★★★☆☆)


```lua

```

How to accumulate elements of a vector (X) to an array (F) based on an index list (I) ? (★★★☆☆)




```lua
X = th.LongTensor{1,2,3,4,5,6}
I = th.LongTensor{1,3,9,3,4,1}
F = th.gather(I,1,X)
print(F)
```

Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★★☆☆)




```lua

```

Considering a four dimensions array, how to get sum over the last two axis at once ? (★★★☆☆)




```lua
A = th.rand(3,4,3,4)
S = A:reshape(3,4,12):sum(3)
print(S)


```

Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices ? (★★★☆☆)




```lua


```

How to get the diagonal of a dot product ? (★★★☆☆)


```lua
A = th.rand(2,3)
B = th.rand(3,2)
print((A*B):diag():sum())

```

Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value ? (★★★☆☆)




```lua

```

Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5) ? (★★★☆☆)


```lua
A = th.rand(5,5,3)
B = th.rand(5,5,1)
B = B:expandAs(A)
C = A:cmul(B)
print(#C)
```

How to swap two rows of an array ? (★★★☆☆)


```lua
A = th.rand(5,3)
print(A)
rows = A:index(1,th.LongTensor{1,3})
A:indexCopy(1,th.LongTensor{3,1},rows)
print(A)
```

Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles (★★★☆☆)


```lua

```

Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C ? (★★★☆☆)


```lua

```

How to compute averages using a sliding window over an array ? (★★★☆☆)


```lua
x = th.range(1,7)
print(x:unfold(1, 3, 1):mean(2))
```

Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★☆☆)




```lua
x = th.range(1,7)
print(x:unfold(1, 3, 1))
```

How to negate a boolean, or to change the sign of a float inplace ? (★★★☆☆)


```lua

```

Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i]) ? (★★★☆☆)


```lua

```

Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i]) ? (★★★☆☆)


```lua



```

Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary) (★★★☆☆)


```lua

```

Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] ? (★★★☆☆)




```lua
x = th.range(1,14)
print(x:unfold(1, 4, 1))
```

Compute a matrix rank (★★★☆☆)


```lua
A = torch.rand(10,10)
A[3] = A[1]*2.0 + A[2]*3.0
_,S,_=torch.svd(A)
rank = S:gt(1e-10):sum()
print(rank)
```

Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★☆☆)


```lua

```

Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★☆☆)


```lua
--function symmetric(Z):
--    return Z + Z:t() - th.diag(Z:diag())
```

Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once ? (result has shape (n,1)) (★★★☆☆)




```lua
p, n = 10, 20
M = th.ones(p,n,n)
V = th.ones(p,n,1)
print(M*V)
```

Consider a 16x16 array, how to get the block-sum (block size is 4x4) ? (★★★☆☆)


```lua

```

How to implement the Game of Life using numpy arrays ? (★★★☆☆)


```lua

```

Given an arbitrary number of vectors, build the cartesian product (every combinations of every item) (★★★☆☆)


```lua

```

How to create a record array from a regular array ? (★★★☆☆)



Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★☆☆)




```lua
Z = th.rand(10)
print(th.pow(Z,3))
print(th.cmul(th.cmul(Z,Z),Z))
```

Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B ? (★★★★☆)



Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★★☆)



Convert a vector of ints into a matrix binary representation (★★★★☆)



Given a two dimensional array, how to extract unique rows ? (★★★★☆)



Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★★☆)



Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★★★) ?




```lua

```
