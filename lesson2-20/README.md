# Coding Cross-Entropy

The formula for calculating cross entropy for two classes is
$-\sum_{i=1}^{m} y_iln(p_i)+(1-y_i)ln(1-p_i)$

Let's code the formula for the Cross-Entropy function in Python.

Write a function that takes as input two lists $Y$, $P$, and returns the float corresponding to their cross-entropy.

> keep in mind that this formula can scale if the given classes are more than two it is tranformed to:
> $-\sum_{i=1}^{n}\sum_{j=1}^{m} y_{ij}ln(p_{ij})$
