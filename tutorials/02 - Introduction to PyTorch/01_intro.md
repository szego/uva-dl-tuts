This is an RMarkdown notebook. I’ll probably do all of the tutorials in
these (using the reticulate R package). No reason in particular other
than I’m really comfortable with R/RStudio. And I will probably cheat
here and there by doing some data processing or plotting in R instead of
Python (but I do want to get as used to Polars as I can).

I should make sure I have the main python libs installed.

``` python
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()

## Progress bar
from tqdm.notebook import tqdm
```

Yup, looks good.

# The basics of PyTorch

``` python
import torch
print("Using torch", torch.__version__)
```

    ## Using torch 2.0.1

# Tensors

``` python
x = torch.Tensor(2, 3, 4)
x
```

    ## tensor([[[0., 0., 0., 0.],
    ##          [0., 0., 0., 0.],
    ##          [0., 0., 0., 0.]],
    ## 
    ##         [[0., 0., 0., 0.],
    ##          [0., 0., 0., 0.],
    ##          [0., 0., 0., 0.]]])

Instantiating a tensor like this allocates memory to it but doesn’t
clear away what was there before. So it will contain whatever junk data
happened to be there.

There are some broadcast functions that can be used instead to fill it
with some initial values:

-   `torch.zeros` Creates a tensor filled with zeros
-   `torch.ones` Creates a tensor filled with ones
-   `torch.rand` Creates a tensor with draws from Uniform(0,1)
-   `torch.randn` Creates a tensor with random draws from Normal(0,1)
-   `torch.arange` Creates a tensor containing sequential values
-   `torch.Tensor (input list)` Creates a tensor from the list elements
    you provide

`torch.arange` has a few different options:

``` python
torch.arange(5)  # [0, 1, 2, 3, 4]
```

    ## tensor([0, 1, 2, 3, 4])

``` python
torch.arange(5, 10)  # [5, 6, 7, 8, 9]
```

    ## tensor([5, 6, 7, 8, 9])

``` python
torch.arange(5, 10, 0.5)  # [5, 5.5, 6, 6.5, ..., 9.5]
```

    ## tensor([5.0000, 5.5000, 6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000, 9.0000,
    ##         9.5000])

They all exclude the top end of the range. That’ll take some getting
used to.

Get the tensor’s shape:

``` python
# These are equivalent
x.shape
```

    ## torch.Size([2, 3, 4])

``` python
x.size()
```

    ## torch.Size([2, 3, 4])

``` python
a, b, c = x.shape
a
```

    ## 2

``` python
b
```

    ## 3

``` python
c
```

    ## 4

We can access a single dimension like so:

``` python
x.shape[0]
```

    ## 2

And we can get the number of dimensions like so:

``` python
len(x.shape)
```

    ## 3

## Tensors to and from numpy

``` python
np_arr = np.array([[1, 2], [3, 4]])
x = torch.from_numpy(np_arr)

# Tensor from numpy array
x
```

    ## tensor([[1, 2],
    ##         [3, 4]], dtype=torch.int32)

``` python
# back to a numpy array
x.numpy()
```

    ## array([[1, 2],
    ##        [3, 4]])

From the tutorial:

> The conversion of tensors to numpy require the tensor to be on the
> CPU, and not the GPU (more on GPU support in a later section). In case
> you have a tensor on GPU, you need to call .cpu() on the tensor
> beforehand. Hence, you get a line like np_arr = tensor.cpu().numpy().

## Operations

Adding and adding-in-place:

``` python
x = torch.rand(2, 3)
y = torch.rand(2, 3)

# does not modify x
x+y
```

    ## tensor([[1.6622, 1.5972, 0.9841],
    ##         [0.8812, 1.5139, 1.7274]])

``` python
# adds y to x in place
# effectively the same as x = x + y
x.add_(y)
```

    ## tensor([[1.6622, 1.5972, 0.9841],
    ##         [0.8812, 1.5139, 1.7274]])

``` python
# Now x is different
x
```

    ## tensor([[1.6622, 1.5972, 0.9841],
    ##         [0.8812, 1.5139, 1.7274]])

> In-place operations are usually marked with a underscore postfix
> (e.g. “add\_” instead of “add”).

Changing the shape of a tensor:

``` python
x.view(3, 2)
```

    ## tensor([[1.6622, 1.5972],
    ##         [0.9841, 0.8812],
    ##         [1.5139, 1.7274]])

Looks like it proceeds by row instead of by column.

``` python
y = torch.arange(6)
y.view(2,3)
```

    ## tensor([[0, 1, 2],
    ##         [3, 4, 5]])

Indeed!

Permuting dimensions:

``` python
x = torch.rand(3, 4, 5)
x.size()
```

    ## torch.Size([3, 4, 5])

``` python
x.permute(2, 0, 1).size()
```

    ## torch.Size([5, 3, 4])

`permute()` must be given the same number of dimensions as the tensor.
For example, `x.permute(2, 0)` returns an error.

### Matrix math

Torch has a bunch of optimized functions for matrix math:

-   `torch.matmul` Performs the matrix product over two tensors, where
    the specific behavior depends on the dimensions. If both inputs are
    matrices (2-dimensional tensors), it performs the standard matrix
    product. For higher dimensional inputs, the function supports
    broadcasting (for details see the documentation). Can also be
    written as `a @ b`, similar to numpy.

-   `torch.mm` Performs the matrix product over two matrices, but
    doesn’t support broadcasting.

-   `torch.bmm` Performs the matrix product with a support batch
    dimension. If the first tensor is of shape (b, n, m), and the second
    tensor (b, m, p), the output is of shape (b, n, p). Basically it
    uses the first dimension as an index and multiplies the matrices of
    dimension (n,m) from the first argument by the matrices of dimension
    (n,p) from the second argument at each index.

-   `torch.einsum` Performs matrix multiplications and more (i.e. sums
    of products) using the Einstein summation convention.

Hah `torch.einsum` sounds neat. Let’s try getting the trace of a matrix.

``` python
x = torch.arange(9).view(3, 3)
torch.einsum("ii", x)
```

    ## tensor(12)

The usual matrix product:

``` python
x = torch.rand(3,3)
y = torch.rand(3,3)

# are equivalent
torch.matmul(x, y)
```

    ## tensor([[0.2503, 0.3791, 0.5554],
    ##         [0.3158, 0.6698, 0.9732],
    ##         [0.4470, 0.2979, 0.3973]])

``` python
torch.einsum("ij,jk", x, y)
```

    ## tensor([[0.2503, 0.3791, 0.5554],
    ##         [0.3158, 0.6698, 0.9732],
    ##         [0.4470, 0.2979, 0.3973]])

Commuting x and y:

``` python
torch.matmul(y, x)
```

    ## tensor([[0.6782, 1.0339, 0.1904],
    ##         [0.4633, 0.4909, 0.0950],
    ##         [0.4427, 0.7118, 0.1484]])

``` python
torch.einsum("ik,ji", x, y)
```

    ## tensor([[0.6782, 1.0339, 0.1904],
    ##         [0.4633, 0.4909, 0.0950],
    ##         [0.4427, 0.7118, 0.1484]])

This is probably most useful when x and y have different dimensions. For
example, we can treat any dimension as a batch dimension:

``` python
z = torch.rand(3, 3, 5)
ans = torch.einsum("ij,jkl", x, z)  # third dim is batch

# same shape as z
ans.size()
```

    ## torch.Size([3, 3, 5])

``` python
# these are equal
ans[:,:,2]
```

    ## tensor([[0.3685, 0.3777, 0.4583],
    ##         [0.5381, 0.4880, 0.8487],
    ##         [0.4349, 0.6441, 0.1765]])

``` python
torch.mm(x, z[:,:,2])
```

    ## tensor([[0.3685, 0.3777, 0.4583],
    ##         [0.5381, 0.4880, 0.8487],
    ##         [0.4349, 0.6441, 0.1765]])

Sick.

Can we use it to implement determinants?

``` python
x = torch.randn(9).view(3, 3)

# build the Levi-Civita tensor
levi_civita = torch.zeros(3, 3, 3)
levi_civita[0, 1, 2] = 1
levi_civita[1, 2, 0] = 1
levi_civita[2, 0, 1] = 1
levi_civita[2, 1, 0] = -1
levi_civita[1, 0, 2] = -1
levi_civita[0, 2, 1] = -1

levi_civita
```

    ## tensor([[[ 0.,  0.,  0.],
    ##          [ 0.,  0.,  1.],
    ##          [ 0., -1.,  0.]],
    ## 
    ##         [[ 0.,  0., -1.],
    ##          [ 0.,  0.,  0.],
    ##          [ 1.,  0.,  0.]],
    ## 
    ##         [[ 0.,  1.,  0.],
    ##          [-1.,  0.,  0.],
    ##          [ 0.,  0.,  0.]]])

``` python
# these are equal
x.det()
```

    ## tensor(7.8653)

``` python
torch.einsum("ijk,i,j,k", levi_civita, x[0,], x[1,], x[2,])
```

    ## tensor(7.8653)

Ooo let me take the cross product of the first two rows of x.

``` python
# these are equal
torch.linalg.cross(x[0,], x[1,])
```

    ## tensor([ 4.0857,  2.0896, -3.0439])

``` python
torch.einsum("ijk,i,j", levi_civita, x[0,], x[1,])
```

    ## tensor([ 4.0857,  2.0896, -3.0439])

I wonder what kinds of situations would require `einsum()` instead of
just using the built-in common operations.

## Other indexing

It’ll take me a while to remember this indexing stuff:

``` python
x = torch.arange(12).view(3, 4)

x
```

    ## tensor([[ 0,  1,  2,  3],
    ##         [ 4,  5,  6,  7],
    ##         [ 8,  9, 10, 11]])

``` python
x[:, 1]   # Second column
```

    ## tensor([1, 5, 9])

``` python
x[0]      # First row
```

    ## tensor([0, 1, 2, 3])

``` python
x[:2, -1] # First two rows, last column
```

    ## tensor([3, 7])

``` python
x[1:3, :] # Middle two rows
```

    ## tensor([[ 4,  5,  6,  7],
    ##         [ 8,  9, 10, 11]])
