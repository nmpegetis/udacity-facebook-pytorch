# Coding Cross-Entropy

The formula for calculating cross entropy is
$-\sum_{i=1}^{m} y_iln(p_i)+(1-y_i)ln(1-p_i)$

Let's code the formula for the Cross-Entropy function in Python.

Write a function that takes as input two lists $Y$, $P$, and returns the float corresponding to their cross-entropy.