# symbolic-time-series

## Search strategy

### Notation
|Symbol|Shape|Description|
|---|---|---|
|$n$|$1$|The number of building blocks|
|$k$|$1$|The beam width|
|$b$|$1$|The number of beams|
|$s$|$1$|The number of predefined useful combinations|
|$\mathbf{F}$|$1\times n$|The matrix representing the building blocks.|
|$\Lambda$|$2^n\times n$|The total number of potential equations.|
|$\mathbf{B}_i$|$bkn\times n$|The potential equations at depth $i$, note that the shape represents an upper bound and is usually smaller due to duplicates.|
|$l$|$k$|A vector of size $k$ which holds the indexes of the rows of $\mathbf{B}_i$ which produce the lowest losses.|
|$\mathbf{B}_i^{l}$|$k\times n$|A subset of $\mathbf{B}_i$ containing only the rows denoted in $l$.|
|$\mathbf{S}$|$s\times n$|A matrix containing predefined useful combination of building blocks.|
|$\mathbf{I}$|$n\times n$|Identity matrix.|
|$\mathbf{I}'$|$(n+s)\times n$|$\begin{bmatrix}\mathbf{I}\\\\\mathbf{S}\end{bmatrix}$|
|$\oslash$|None|This denotes a special operation between two matrices $\mathbf{X}_1$ and $\mathbf{X}_2$, which are required to have the same shape in dimension 1. It represents:<br>$\mathbf{X}_1\oslash\mathbf{X}_2=\begin{bmatrix}\mathbf{X}_1^0\mid\mathbf{X}_2\\\\\dots\\\\\mathbf{X}_1^n\mid\mathbf{X}_2\end{bmatrix}$|

### Beam search

The matrix of potential equations at depth $d$ can be defined as follows:

$$
    \mathbf{B}_{d}=
    \begin{cases}
      \mathbf{I}', & \text{if}\ d=0 \\
      \mathbf{B}_{d-1}^{\argmin_k(\mathcal{L}(\mathbf{B}_{d-1} \times \mathbf{F}))} \oslash \mathbf{I}', & \text{otherwise}
    \end{cases}
$$

To illustrate to the process of calculating the matrix $\mathbf{B}_d$ consider the following example:
$$
k = 2
$$

$$
\mathbf{S}=\begin{bmatrix}
    0 & 0 & 1 & 1 & 0 & 0
\end{bmatrix}
$$

$$
\mathbf{F}=\begin{bmatrix}
    x_1 & x_2 & x_1^2 & x_2^2 & e^{x_1} & e^{x_2} \\
\end{bmatrix}
$$

As stated in Equation~\ref{eq:beam_search} $\mathbf{B}_0$ is defined as follows:

$$
\mathbf{B}_0=\mathbf{I}'=
\begin{bmatrix}
    \mathbf{I}\\
    \mathbf{S}
\end{bmatrix}=
\begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 1 & 0 & 0 
\end{bmatrix}
$$

To calculate $\mathbf{B}_1$, the losses for all potential equations in $\mathbf{B}_0$ need to be calculated.
Which is done as follows:
$$
\mathcal{L}(\mathbf{B}_0 \times \mathbf{F})=\mathcal{L}(
\begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 1 & 0 & 0 
\end{bmatrix} \times 
\begin{bmatrix}
    x_1 & x_2 & x_1^2 & x_2^2 & e^{x_1} & e^{x_2} \\
\end{bmatrix}
)=
\mathcal{L}(
\begin{bmatrix}
    x_1 \\
    x_2 \\
    x_1^2\\
    x_2^2 \\
    e^{x_1} \\
    e^{x_2} \\
    x_1^2 + x_2^2
\end{bmatrix}
)=
\begin{bmatrix}
    0.25 \\
    0.45 \\
    0.10 \\
    0.45 \\
    0.67 \\
    0.50 \\
    0.45
\end{bmatrix}
$$

Then, the indexes for the $k$ smallest losses of $\mathbf{B}_0$ are calculated.
Subsequently, the rows from $\mathbf{B}_0$ are selected.
$$
l=\argmin_k(\mathcal{L}(\mathbf{B}_0 \times \mathbf{F}))=\begin{bmatrix}
    0 & 2 
\end{bmatrix}
$$

$$
\mathbf{B}_0^{\argmin_k(\mathcal{L}(\mathbf{B}_0 \times \mathbf{F}))}=
\mathbf{B}_0^{l}=
\begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 
\end{bmatrix}
$$

Finally $\mathbf{B}_1$ can be calculated:
$$
\mathbf{B}_1=\mathbf{B}_0^{l} \oslash \mathbf{I}'=\begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 
\end{bmatrix} \oslash \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 1 & 1 & 0 & 0 
\end{bmatrix}=
\begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    1 & 1 & 0 & 0 & 0 & 0 \\
    1 & 0 & 1 & 0 & 0 & 0 \\
    1 & 0 & 0 & 1 & 0 & 0 \\
    1 & 0 & 0 & 0 & 1 & 0 \\
    1 & 0 & 0 & 0 & 0 & 1 \\
    1 & 0 & 1 & 1 & 0 & 0 \\
    
    1 & 0 & 1 & 0 & 0 & 0 \\
    0 & 1 & 1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 & 1 & 0 \\
    0 & 0 & 1 & 0 & 0 & 1 \\
    0 & 0 & 1 & 1 & 0 & 0 
\end{bmatrix}
$$

## Extention to time series forecasting

The base case deals with the classical regression task. 
While any time series forecasting task can be translated into a classic 
regression task by treating the input time series as a feature vector, we
lose the additional information encoded in the data.
Therefore we propose the following components to encode the time dependent
features in the equation discovery framework.

We require the following attributes:
- $\Delta^-$, which denotes the maximum of historical time points the system 
is allowed to use.
- $\Delta^+$, which denotes the target for the prediction.

The task of equation discovery for time series forcasting than becomes the
following:
$$
f^* \in \argmin\limits_{f \in \Lambda} g(f(X_{features,\Delta^-}),
Y_{target,\Delta^+})
$$
Where:
- $\Lambda$, defines the complete search space of possible equations.
- $X_{features,\Delta^-}$, defines the matrix of input features, with a maximum
historic horizon of $\Delta^-$.
- $Y_{target,\Delta^+}$, defines the vector of target values for the feature
$target$ at time point $\Delta^+$.
- $g$, defines the loss function.
- $f$, defines a function in the search space $\Lambda$.
- $f^*$, defines the equation $f$ in $\Lambda$ that minimises $g$ for 
$X_{features,\Delta^-}$ and $Y_{target,\Delta^+}$.

### Example case

Imagine we have a dataset of a cycling athlete containing two features: heart rate (bpm), and power (watts).

| Time (t) | Heart Rate (bpm) | Power (W) |
| -------- | ---------------- | --------- |
| 1        | 110              | 120       |
| 2        | 118              | 150       |
| 3        | 128              | 190       |
| 4        | 138              | 220       |
| 5        | 150              | 250       |
| 6        | 158              | 260       |
| 7        | 160              | 200       |
| 8        | 156              | 160       |
| 9        | 150              | 140       |
| 10       | 145              | 130       |

The task is to predict the power output at time $t=11$ based on the previous values of heart rate and power.
To achieve this, we can define the following parameters:
- $\Delta^-$ = 3, meaning we can use the last 3 time points
- $\Delta^+$ = 1, meaning we want to predict the power output at time $t=11$.

The input feature matrix for $X_{features,3}^8$ would look like this:
$$
X_{features,8} =
\begin{bmatrix}
    150 & 250 \\
    158 & 260 \\
    160 & 200 \\
\end{bmatrix}
$$
The target vector for $Y_{target,1}^8$ would look like this:
$$
Y_{target,1}^8 =
\begin{bmatrix}
    156 \\
\end{bmatrix}
$$

In total we have 8 training samples, which can be used to train the equation discovery system.
With the goal of finding an equation that using $X_{features,3}^{11}$ predicts $Y_{target,1}^{11}$.
With the following input feature matrix:
$$
X_{features,3}^{11} =
\begin{bmatrix}
    160 & 200 \\
    156 & 160 \\
    150 & 140 \\
\end{bmatrix}
$$

#### Convential regression

If you would approach this task from a conventional regression perspective, you would change the input feature matrix into a vector of features.
$$
X_{features,3}^{11} = X_t =
\begin{bmatrix}
    160 & 200 & 156 & 160 & 150 & 140 \\
\end{bmatrix}
$$

A potential equation could be:
$$
f(X_{features,3}^{11}) = 0.7 \cdot X_{f}^4 + 0.1 \cdot X_{f}^5 + 10
$$
$$
f(X_{features,3}^{11}) = 0.7 \cdot 200^4 + 0.1 \cdot 140^5 + 10
$$
$$
f(X_{features,3}^{11}) = 156
$$




