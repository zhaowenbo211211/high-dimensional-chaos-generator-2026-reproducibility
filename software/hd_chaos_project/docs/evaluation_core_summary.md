# Current Core Content Summary for `\subsection{Evaluation Objectives and Compared 4D Chaotic Systems}`

This markdown file consolidates the **current completed core content** of the paper so that the subsequent discussion on the evaluation section can proceed on a stable basis.

---

## 1. Base high-dimensional delayed discrete-time chaotic system

The starting point of the paper is the following $n$-dimensional delayed discrete-time system on $[0,1)^n$ (mod 1):

\[
\begin{cases}
x_1(i+1)
= a_{1,1}\,x_1(i)
+ a_{1,n}\,x_n\bigl(i-(n-2)\bigr),\\[0.5mm]
x_2(i+1)
= a_{2,1}\,x_2(i)
+ a_{2,n}\,x_n\bigl(i-(n-3)\bigr),\\[0.5mm]
\hspace{16mm}\vdots\\[0.5mm]
x_{n-1}(i+1)
= a_{n-1,1}\,x_{n-1}(i)
+ a_{n-1,n}\,x_n(i),\\[1mm]
x_n(i+1)
= b\,x_{n-2}(i)
+ 4\bigl(x_{n-1}(i)+x_n(i)\bigr)
\Bigl(1-\bigl(x_{n-1}(i)+x_n(i)\bigr)\Bigr),
\end{cases}
\]

where $a_{j,1},a_{j,n}\in\mathbb{Z}_{>0}$ and $b\in\mathbb{Z}_{>0}$ are integer parameters.

### Structural interpretation
- The first $n-1$ coordinates are **linear delayed channels**.
- The last coordinate $x_n$ is the **nonlinear core**, driven by a logistic-type term and a feedback term $b\,x_{n-2}(i)$.
- The whole system is designed over modular arithmetic and is intended for fixed-point digital implementation.

### Role in the paper
This system is the **continuous/discrete mathematical prototype** from which the later 4D realizations, parameter synthesis, digital implementation, and comparison study are developed.

---

## 2. Integer parameter synthesis: R1--LY algorithm

The paper proposes **Algorithm R1--LY**, a one-shot integer synthesis framework for generating valid parameter sets.

### Main inputs
- Dimension $n\ge 3$
- Base odd gain $A_{\mathrm{odd}}$
- Base even gain $B_{\mathrm{even}}$
- Step size $\Delta=2$
- Safety ratio $\eta$
- Logistic-coupling safety factor $\kappa$
- Band-width factor $\tau$

### Main mechanism
1. Construct complementary parity-alternating sequences
   - $a_{j,1} = A_{\mathrm{odd}} + (j-1)\Delta$
   - $a_{j,n} = B_{\mathrm{even}} - (j-1)\Delta$
2. Compute
   - $A_{\min} = A_{\mathrm{odd}}$
   - $\beta = \left(\sum_{j=1}^{n-1} a_{j,n}^2\right)^{1/2}$
3. Set
   - $b = \max\bigl(2,\lfloor \eta A_{\min}\rfloor\bigr)$
   - $c = \sqrt{b^2+32}$
4. Enforce the cone-field condition
   \[
   (A_{\min}-1)^2 > 4\beta c
   \]
5. Check no-wrap budget and band-width constraints
6. Return the validated parameter set
   \[
   \mathcal{P}=(\{a_{j,1}\}_{j=1}^{n-1},\ \{a_{j,n}\}_{j=1}^{n-1},\ b)
   \]

### Role in the paper
This algorithm is the **bridge between the theory and the realizable digital instances**. It turns the high-dimensional model into reproducible integer parameter sets without ad hoc numerical tuning.

---

## 3. 4D instantiation and golden range

The paper then specializes the theory to the **4D case**.

### Fixed coupling vector
For $n=4$, the coupling coefficients are fixed as

\[
\{a_{j,n}\} = [50,48,46].
\]

### Golden working range
Substituting these values into Condition (S) and combining them with the 16-bit implementation constraints yields the one-dimensional valid range

\[
\boxed{271 \le a_{1,1} \le 507}.
\]

### Meaning of the two bounds
- **Lower bound 271**: ensures
  \[
  (A_{\min}-1)^2 > 4\beta c,
  \]
  so that all physical Lyapunov exponents remain strongly positive.
- **Upper bound 507**: satisfies the overflow-safe 16-bit implementation constraint
  \[
  a_{j,1}\le 511.
  \]

### Role in the paper
This golden range is crucial because it turns the 4D design problem into a **systematic one-dimensional validation problem**, rather than an arbitrary multi-parameter random search.

---

## 4. Dynamics enhancer under finite precision

To improve digital behavior under finite precision, the paper introduces a **bit-rotation based dynamics enhancer**.

### Enhancer principle
After one iteration of the base map $G$, the $n$ physical coordinates are:
1. packed into one $nw$-bit word,
2. cyclically rotated left by $k_t$ bits,
3. unpacked back into the $n$ coordinates.

### Current final schedule
The current final adopted step schedule is

\[
k_t = 1 + (t \bmod (w-1)).
\]

For the 16-bit case, this becomes

\[
k_t = 1,2,3,\dots,15,1,2,3,\dots
\]

### Algorithm form
The final enhancer is summarized as:
1. $x \gets G(x)$
2. $k_t \gets 1 + (t \bmod (w-1))$
3. $Z \gets \Pi(x)$
4. $Z \gets \mathrm{ROL}_{nw}(Z,k_t)$
5. $x \gets \Pi^{-1}(Z)$

### Why this version is retained
During implementation and testing, several pseudo-random step-generation variants were explored, including LFSR-based schedules. However, after correcting the Wolf Lyapunov implementation, the **simple cyclic schedule**:
- achieved performance very close to or better than the pseudo-randomized versions,
- was more stable in Lyapunov-based tests,
- and remained much simpler for hardware realization.

### Hardware interpretation
This final enhancer only requires:
- one $nw$-bit cyclic shifter,
- wiring for pack/unpack,
- and a small modulo-$(w-1)$ counter.

It does **not** require extra pseudo-random logic such as LFSR feedback circuitry in the final adopted design.

---

## 5. Current practical comparison conclusion

At the present stage, the core completed work is:
- a mathematically defined **high-dimensional delayed chaotic map**;
- a **theory-driven integer parameter synthesis method** (R1--LY);
- a **4D instantiation with closed-form valid parameter range**;
- and a **finite-precision dynamics enhancer** based on cyclic bit rotation.

These are the **four core completed components** that should serve as the reference point for revising

`\subsection{Evaluation Objectives and Compared 4D Chaotic Systems}`

and for building the subsequent experimental comparison section.

---

## 6. Suggested use of this summary in the next interaction

Based on this consolidated core content, the next steps can now focus on:
1. rewriting the opening paragraphs of `\subsection{Evaluation Objectives and Compared 4D Chaotic Systems}`;
2. clarifying what exactly is being compared:
   - R1--LY-generated 4D instances,
   - their enhanced versions,
   - and existing 4D digital chaotic maps;
3. defining the fairness criteria:
   - same 4D setting,
   - same 16-bit fixed-point arithmetic,
   - same trajectory length,
   - same initial-value/parameter selection principles,
   - Python simulation only,
   - FPGA-oriented complexity analysis only;
4. organizing the comparison metrics:
   - Lyapunov spectrum,
   - complexity,
   - distribution uniformity,
   - state-space coverage,
   - initial-value sensitivity,
   - cycle length / finite-precision degradation.
