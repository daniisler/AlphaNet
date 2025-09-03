# Flexible ZBL Potential (Ziegler–Biersack–Littmark Potential)

## 1. Basic Idea

The ZBL potential is a **screened Coulomb potential** widely used to describe short-range interactions between nuclei, especially in ion–solid collisions and radiation damage simulations.

The bare Coulomb interaction between two nuclei is:

$$
V_{\text{Coul}}(r) = \frac{Z_1 Z_2 e^2}{4\pi \epsilon_0  r}
$$

Due to the screening effect of the electron clouds, the actual interaction is reduced by a screening function $\phi(x)$:

$$
V(r) = \frac{Z_1 Z_2 e^2}{4\pi \epsilon_0  r}  \phi\left(\frac{r}{a}\right)
$$

where:  
- $Z_1, Z_2$ are the atomic numbers,  
- $r$ is the interatomic distance,  
- $a$ is the screening length,  
- $\phi(x)$ is a dimensionless screening function satisfying $\phi(0)=1$ and monotonically decreasing with $x$.  

---

## 2. Screening Length

In the original ZBL potential, the screening length is defined as:

$$
a = 0.8854  a_0  (Z_1^{0.23} + Z_2^{0.23})^{-1}
$$

where:  
- $a_0 = 0.529177  \text{Å}$ is the Bohr radius.  

In a generalized form, adjustable parameters $\gamma, \alpha$ can be introduced:

$$
a = \gamma \cdot 0.8854  a_0  (Z_1^\alpha + Z_2^\alpha)^{-1}, 
\quad \gamma > 0,  \alpha > 0
$$

---

## 3. Screening Function

The original ZBL screening function is expressed as a weighted sum of four exponential terms:

$$
\phi(x) = \sum_{i=1}^4 w_i  \exp(-b_i x)
$$

with parameters:  

$$
\begin{aligned}
w &= (0.1818,  0.5099,  0.2802,  0.02817) \\
b &= (3.2,  0.9423,  0.4029,  0.2016)
\end{aligned}
$$

These weights satisfy $\sum_i w_i = 1$, ensuring $\phi(0)=1$, so that the potential recovers the correct bare Coulomb limit as $r \to 0$.

---

## 4. Generalized Form (fZBL)

A more flexible version, the **fZBL potential**, extends the ZBL form:

$$
V(r) = \frac{Z_1 Z_2 e^2}{4\pi \epsilon_0  r}  \phi\left(\frac{r}{a}\right)
$$

with  
- adjustable screening length:  

$$
a = \gamma \cdot 0.8854  a_0 (Z_1^\alpha + Z_2^\alpha)^{-1}
$$

- generalized screening function:  

$$
\phi(x) = \sum_{i=1}^M w_i  \exp(-b_i x^p)
$$

where:  
- $M$: number of terms (default $M=4$, can be extended to 8 or more),  
- $p > 0$: exponent power (default $p=1$),  
- $w_i \geq 0,  \sum_i w_i = 1$,  
- $b_i > 0$.  

This allows fitting $\{w_i, b_i, \gamma, \alpha\}$ to ab initio or experimental data.  

**We set M=8 and fit it on 6520 element pairs, calculated by VASP.**

---

## 5. Units and Constants

In atomistic simulations, it is convenient to work in **eV and Å** units:

$$
\frac{e^2}{4\pi \epsilon_0} = 14.399645  \text{eV·Å}
$$

Thus:  
- $r$ is measured in Å,  
- $V(r)$ is expressed in eV.  

---

## 6. Key Properties

- $\phi(0) = 1$: ensures the correct nuclear–nuclear Coulomb limit as $r \to 0$.  
- $\phi(x)$ is monotonically decreasing: guarantees screening increases with distance.  
- Flexible fitting: generalized ZBL (fZBL) can adapt to different materials (metals, alloys, molecular systems).  

---