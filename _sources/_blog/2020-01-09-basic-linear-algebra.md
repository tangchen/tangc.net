---
title: "Some Basic Linear Algebra"
date: 2020-01-09
categories: ["Math and Stats"]
tags: ["linear algebra"]
math: true
draft: false
---

# Some Basic Linear Algebra

Purely out of curiosity, I'm recently reading [the deep learning book](http://www.deeplearningbook.org/) (Goodfellow, Bengio, & Courville, 2016). I noticed that Chapter 2 Linear Algebra is a very quick and effective summary of linear algebra. It provides the right amount of linear algebra for deep learning, as well as machine learning and statistics needed for my research. Below are excerpts from the deep learning book all credit goes to Dr. Ian Goodfellow and his colleagues.

<!--more-->

## Scalars, Vectors, Matrices and Tensors

- scalar: just a single number
- vector: an array of numbers
- matrix: a 2-dimensional array of numbers
- tensor: an n-dimensional array of number

## Transpose

The **transpose** of a matrix is the mirror image of the matrix across the main diagonal:

$$A_{i,j}^{\top} = A_{j,i}$$

Transpose of a scalar is itself. Transpose of a column vector is a row vector and vice versa.

## Multiplication

### Matrix product

The matrix product of matrices $A$ and $B$ is a third matrix $C$. In order for this product to be defined, $A$ must have the same number of columns as $B$ has rows. If $A$ is of shape $m \times n$ and $B$ is of shape $n \times p$, then $C$ is of shape $m \times p$.

$$C_{i, j} = \sum_k A_{i, k} B_{k, j}$$

### Element-wise product (aka Hadamard product)

$$A \odot B$$

### Dot product

The dot product between two vectors $x$ and $y$ of the same length is the matrix product $x^\top y$. the matrix product $C$ can be thought of as the dot products of each corresponding row in $A$ and column in $B$.

### Properties of matrix multiplication

- Distributive: $A(B + C)=AB + AC$
- Associative: $A(BC) = (AB)C$
- Not commutative: $AB = BA$ is not always true
- But dot product between two vectors is commutative: $x^\top y = y^\top x$
- Transpose of a matrix product: $(AB)^\top = B^\top A^\top$ (this can be used to prove $x^\top y = y^\top x$)

## Identity and Inverse Matrices

Identity matrix: denoted as $I_n$, all its entries along the main diagonal are 1, all the others are zero.

The inverse of $A$, denoted as $A^{-1}$, is defined as the matrix such that

$$A^{-1}A = I_n$$

## Linear Dependence and Span

A **linear combination** of some set of vectors $\{v^{(1)}, ... , v^{(n)}\}$ is given by multiplying
each vector $v^{(i)}$ by a corresponding scalar coefficient and adding the results:

$$\sum_i c_iv^{(i)}$$

The **span** of a set of vectors is the set of all points obtainable by linear combination of the original vectors.

Determining whether $Ax = b$ has a solution thus amounts to testing whether b is in the span of the columns of $A$. This particular span is known as the **column space** or the **range** of $A$.

## Norms

Norm is the measure of the size of a vector. The $L^p$ norm is defined as:

$$\|x\|_p = \Big(\sum_i |x_i|^p\Big)^{\frac{1}{p}}$$

The **Euclidean norm**, or $L^2$ norm, is used very frequently. It is also common to use the squared $L^2$ norm, which is simply $x^{\top} x$.

However squared $L^2$ norm is undesirable because it changes very slowly near the origin. So when it is important to discriminate between values that are exactly zero and values that are very small but nonzero, we could use $L^1$ norm:

$$\|x\|_1 = \sum_i|x_i|$$

**Max norm** is also common, which is defined by the absolute value of the element with the largest magnitude in the vector:

$$\|x\|_{\infty} = \max_i|x_i|$$

Above norms describe the size of vectors. **Frobenius norm** can measure the size of a matrix:

$$\|A\|_F = \sqrt{\sum_{i, j} A^2_{i,j}}$$

This is analogous to the $L^2$ norm of a vector.

The dot product of two vectors can be written in terms of norms:

$$x^{\top}y = \|x\|_2\|y\|_2\cos{\theta}$$

where $\theta$ is the angle between $x$ and $y$.

## Special Kinds of Matrices and Vectors

### Diagonal matrix

a matrix $D$ is diagonal if and only if $D_{i, j} = 0$ for all $i \ne j$. We write $\text{diag}(v)$ to denote a square diagonal matrix whose diagonal entries are given by the entries of the vector $v$. 

Diagonal matrices are interesting because:

1. Multiplying by a diagonal matrix is very computationally efficient: $\text{diag}(v) x = v \odot x$

2. Inverting a diagonal matrix is easy: $\text{diag}(v)^{-1} = \text{diag}(1/v_1, ..., 1/v_n)^{\top}$

Diagonal matrices do not need to be square. For a non-square diagonal matrix $D$, the product $Dx$ will involve scaling each element of $x$, and either concatenating some zeros to the result if $D$ is taller than it is wide, or discarding some of the last elements of the vector if $D$ is wider than it is tall.

### Symmetric matrix

A symmetric matrix is any matrix that is equal to its own transpose:

$$A = A^{\top}$$

### Unit vector

A unit vector is a vector with unit norm:

$$\|x\|_2 = 1$$

### Orthogonal vectors and orthogonal matrices

Vector $x$ and vector $y$ are orthogonal to each other if $x^{\top}y = 0$. If the vectors are not only orthogonal but also have unit norm, we call them **orthonormal**.

An orthogonal matrix is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal:

$$A^{\top}A = AA^{\top} = I$$

Orthogonal matrices are of interest because their inverse is very cheap to compute:

$$A^{-1} = A^{\top}$$

Counterintuitively, rows and columns of an orthogonal matrix are not merely orthogonal but fully orthonormal. There is no special term for a matrix whose rows or columns are orthogonal but not orthonormal.

[Definition from Wikipedia](https://en.wikipedia.org/wiki/Orthogonal_matrix): "An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors (i.e., orthonormal vectors)."




*(To be continued...)*