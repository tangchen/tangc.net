<!-- ---
title: "A Very Detailed Bias-Variance Breakdown"
date: 2019-09-01
categories: ["Math and Stats"]
tags: ["Machine learning", "Bias-variance trade-off", "Prediction"]
draft: false
math: true
--- -->

# A Very Detailed Bias-Variance Breakdown

Although the concept of bias-variance trade-off is often discussed in machine learning textbooks, e.g., Bishop (2006), Hastie, Tibshirani, and Friedman (2009), James et al. (2013), I also find it important in almost any occasions in which we need to fit a statistical model on a data set with a limited number of observations. To better understand the trade-off, we should be clear about the bias-variance decomposition. I prefer to call it bias-variance breakdown cause there are fewer syllables. This post is an attempt to go through the breakdown in a very detailed manner, mainly for my future reference. It may not be 100% correct because I'm very new to this topic. I will make changes if there is anything wrong.

## What is being broken down?

First off, the bias-variance concept lives in theory. In practice, we have no way to separate bias and variance. What we could observe is just the sum of them, as well as something called "irreducible error". We will cover that later in this post.

Since we are talking about theory, let's make up the thing that needs to be broken down. Let's assume that we are interested in studying the relationship between $X$ and $Y$. Suppose the relationship is:
$$Y = f(X) + \epsilon$$,

where $\epsilon$ is i.i.d. $E(\epsilon) = 0$ and $\text{Var}(\epsilon) = \sigma^2$.

Then we obtain a data set, $D$, that has $x_i$ and $y_i$ ($D = [x_i, y_i]$). Note that in the data set, $X$ and $Y$ are lowercase and have subscripts. This is because we are referring to observed data of $X$ and $Y$.

As usual, we fit a model to $D$ and obtain the model $\hat{f}(x_i)$. There is a "hat" on $f(x_i)$ because we are estimating the true model, $f()$. After getting $\hat{f}()$, we are interested in how this model would perform in the future. So we have to come up with a way to measure the performance of the model when applied to new data. The most common way to measure the predictive performance of a model is mean squared error (MSE) on a new data set $[x^\ast, y^\ast],$ or theoretically, the expected squared error:

$$E[(y^\ast - \hat{f}(x^\ast))^2]$$

This is the thing to be broken down.

## Breaking it down
Before we go further, we need to make sure we are clear about which is which. Since this is an expected value, there must be random variables in this equation. What is random here? First, let's take a look at $\hat{f}(x^\ast)$.

We know that $\hat{f}()$ comes from $D$, and $D$ contains $\epsilon$, because the true model we assumed is $Y = f(X) + \epsilon$. So $\hat{f}()$ also contains $\epsilon$ and hence it is a random variable. Second, what about $y^\ast$? Since $[x^\ast, y^\ast]$ is a sample from the true model, it again contains $\epsilon$, therefore $y^\ast$ is also a random variable.

Let's now play a mathematical trick:

$$E[(y^\ast - \hat{f}(x^\ast))^2] = E[(y^\ast - f(x^\ast) + f(x^\ast) - \hat{f}(x^\ast))^2]$$

Here we just add and subtract $f(x^\ast)$, nothing is changed. Let $A = y^\ast - f(x^\ast)$ and $B = f(x^\ast) - \hat{f}(x^\ast)$. Then the above equation becomes:

$$\begin{aligned}
E[(A + B)^2] &= E[A^2 + B^2 + 2AB] \\
&= E[A^2] + E[B^2] + 2E[AB]
\end{aligned}$$

Let's put $A$ and $B$ back

$$E[(y^\ast - f(x^\ast))^2] + E[(f(x^\ast) - \hat{f}(x^\ast))^2] + 2E\{[y^\ast - f(x^\ast)][f(x^\ast) - \hat{f}(x^\ast)]\}$$

This is very complicated, especially the long thing on the right. Let's first expand it:
$$2\{E[y^\ast f(x^\ast)] - E[y^\ast \hat{f}(x^\ast)] - E[f(x^\ast) f(x^\ast)] + E[f(x^\ast) \hat{f}(x^\ast)]\}$$

According to our theoretical model, we know that $y^\ast = f(x^\ast) + \epsilon$, so

$$2\{E[(f(x^\ast) + \epsilon) f(x^\ast)] - E[(f(x^\ast) + \epsilon) \hat{f}(x^\ast)] - E[f(x^\ast) f(x^\ast)] + E[f(x^\ast) \hat{f}(x^\ast)]\}$$

$$2\{[f(x^\ast)]^2 - E[f(x^\ast) \hat{f}(x^\ast) + \epsilon \hat{f}(x^\ast)] - [f(x^\ast)]^2 + E[f(x^\ast) \hat{f}(x^\ast)]\}$$

$$2\{[f(x^\ast)]^2 - E[f(x^\ast) \hat{f}(x^\ast)] + E[\epsilon \hat{f}(x^\ast)] - [f(x^\ast)]^2 + E[f(x^\ast) \hat{f}(x^\ast)]\}$$

Four terms cancel out, the term $E[\epsilon \hat{f}(x^\ast)] = 0$, because $\epsilon$ and $\hat{f}(x^\ast)$ are independent. Therefore $E[\epsilon \hat{f}(x^\ast)] = E[\epsilon] \times E[\hat{f}(x^\ast)] = 0$.

OK. That long thing becomes zero and we are left with

$$E[(y^\ast - f(x^\ast))^2] + E[(f(x^\ast) - \hat{f}(x^\ast))^2]$$

Now let's play a similar trick on the second term, the first term remains unchanged.

$$E[(y^\ast - f(x^\ast))^2] + E\{[f(x^\ast) - E[\hat{f}(x^\ast)] + E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]^2\}$$

Again let $A = f(x^\ast) - E[\hat{f}(x^\ast)]$ and $B = E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)$.

$$E[(y^\ast - f(x^\ast))^2] + E[(A + B)^2]$$

$$E[(y^\ast - f(x^\ast))^2] + E[A^2] + E[B^2] + 2E[AB]$$

Plug in $A$ and $B$.

$$E[(y^\ast - f(x^\ast))^2] + E\{[f(x^\ast) - E[\hat{f}(x^\ast)]]^2\} + E\{[E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]^2\}$$
$$ + 2E\{[f(x^\ast) - E[\hat{f}(x^\ast)]][E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]\}$$

Let's again look at the most annoying thing on the second row. We notice that (1) $f(x^\ast) - E[\hat{f}(x^\ast)]$ is a constant, and (2) the expected value of $E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]$ is just $E\{E[\hat{f}(x^\ast)]\} - E[\hat{f}(x^\ast)]$. This equals $E[\hat{f}(x^\ast)] - E[\hat{f}(x^\ast)] = 0$.

So we are left with 

$$E[(y^\ast - f(x^\ast))^2] + E\{[f(x^\ast) - E[\hat{f}(x^\ast)]]^2\} + E\{[E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]^2\}$$

We also notice that, in the second term, both $f(x^\ast)$ and $E[\hat{f}(x^\ast)]$ are constants, so we can drop the expectation operator.

$$E[(y^\ast - f(x^\ast))^2] + [f(x^\ast) - E[\hat{f}(x^\ast)]]^2 + E\{[E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]^2\}$$

Cool! We have finished the breakdown.

## Naming things

As a final step, let's name the three terms in

$$E[(y^\ast - f(x^\ast))^2] + [f(x^\ast) - E[\hat{f}(x^\ast)]]^2 + E\{[E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]^2\}$$

The first term, $E[(y^\ast - f(x^\ast))^2]$, is the variance of $\epsilon$, which is $\sigma^2$. We call this "irreducible error" or "irreducible noise".

The second term, $[f(x^\ast) - E[\hat{f}(x^\ast)]]^2$, is the "squared bias", because the definition of the bias of an estimator is $\text{bias}(\hat{\theta}) = E(\hat{\theta}) - \theta$.

The third term, $E\{[E[\hat{f}(x^\ast)] - \hat{f}(x^\ast)]^2\}$is, by definition, the variance of $\hat{f}(x^\ast)$.

So to put everything together, 

$$E[(y^\ast - \hat{f}(x^\ast))^2] = \text{Var}(\epsilon) + \text{bias}^2(\hat{f}(x^\ast)) + \text{Var}(\hat{f}(x^\ast))$$

## Reference

Bishop, Christopher M. (2006). *Pattern recognition and machine learning*. New York: Springer,

Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. (2013). *An introduction to statistical learning: with applications in R*. New York :Springer,

Hastie, T., Tibshirani, R., & Friedman, J. H. (2009). *The elements of statistical learning: data mining, inference, and prediction*. 2nd ed. New York: Springer.