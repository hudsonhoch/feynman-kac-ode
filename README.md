# Feynman-Kac for ODEs

This repo provides an implementation of the results of the paper [A Feynman-Kac Type Theorem for ODEs: Solutions of Second Order ODEs as Modes of Diffusions](https://arxiv.org/abs/2106.08525).

In its essence, a general second order ordinary differential equation of the form

$$
\ddot{y}(t)+f(t)\dot{y}(t)+g(t)y(t)+h(t)=0
$$

where $y(0)=0$ and $y(T)=a$, can be solved by finding the most likely path of a particular diffusion. More precisely, the solution $y(t)$ is

$$
y(t)=e^{-\frac{1}{2}\int_{t}^{T}f(s)ds}\text{Mode}\Big[X(t)\Big|X(T)=a\Big]
$$

where the diffusion $X(t)$ satisfies the stochastic differential equation

$$
dX(t)=\Big[A(t)X(t)+D(t)\Big]dt+dB(t)
$$

where $A(t)$ and $D(t)$ solve a system of first order system of ODEs defined by $g(t)$ and $h(t)$.

For an example on how to use this implementation, see the [example.ipynb](https://github.com/hudsonhoch/feynman-kac-ode/blob/main/example.ipynb) file.
