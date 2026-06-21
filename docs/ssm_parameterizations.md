# S5 Dynamics Parameterizations

## Where The Original S5 Dynamics Live

The original S5 state dynamics are implemented in `s5/ssm.py`.
`S5SSM` owns the diagonal continuous-time eigenvalues `Lambda`, the input and
output maps `B` and `C`, the discretization (`zoh` or `bilinear`), the parallel
scan, and the optional direct feedthrough term `D`.

Training builds the SSM initializer in `s5/train.py`, then passes it unchanged
through the existing `SequenceLayer`, stacked encoder, classifier/retrieval
heads, dataloaders, optimizer, logging, and evaluation loop.

The hardware-friendly alternatives are implemented in
`s5/ssm_parameterizations.py` as a sibling module, `RealValuedSSM`, so the
original S5 code path is preserved.

## Config Options

Use:

```bash
python run_train.py model.ssm_param=original seed=0
python run_train.py model.ssm_param=original_no_D seed=0
python run_train.py model.ssm_param=real_decay seed=0
python run_train.py model.ssm_param=resonant_2x2 seed=0
python run_train.py model.ssm_param=energy_shaped_2x2 seed=0
```

The repository also accepts its original argparse style:

```bash
python run_train.py --ssm_param real_decay --jax_seed 0
```

Allowed values:

```text
original
original_no_D
real_decay
resonant_2x2
energy_shaped_2x2
```

`--use_D true` is honored only for `original`. `original_no_D` and all
hardware-friendly variants force `D = 0`.

## Parameterizations

### original

Uses the original complex diagonal S5 implementation:

```text
x[t+1] = A_bar x[t] + B_bar u[t]
y[t] = C x[t] + D u[t]
```

This is the digital upper bound and may use the original internal complex
representation.

### original_no_D

Uses the original S5 dynamics, but omits the trainable direct feedthrough
parameter:

```text
y[t] = C x[t]
```

### real_decay

Uses real negative continuous-time eigenvalues:

```text
A = -diag(alpha_i)
alpha_i = softplus(raw_alpha_i) + eps
```

For zero-order hold:

```text
lambda_bar_i = exp(-alpha_i dt_i)
B_bar_i = (1 - lambda_bar_i) / alpha_i * B_i
```

`B` and `C` are real-valued, and `D = 0`.

### resonant_2x2

Uses explicit real 2x2 continuous-time blocks:

```text
A_k = [[-alpha_k, -omega_k],
       [ omega_k, -alpha_k]]
alpha_k = softplus(raw_alpha_k) + eps
```

The exact discrete block is:

```text
A_bar_k = exp(-alpha_k dt_k)
          [[cos(omega_k dt_k), -sin(omega_k dt_k)],
           [sin(omega_k dt_k),  cos(omega_k dt_k)]]
```

This is equivalent to complex conjugate eigenvalues
`-alpha_k +/- i omega_k`, but the implementation keeps the scan state, `B`,
and `C` real-valued.

### energy_shaped_2x2

Uses a blockwise energy-shaped realization:

```text
A_k = (J_k - R_k) Q_k
J_k = [[0, -omega_k],
       [omega_k, 0]]
R_k = alpha_k I
Q_k = q_k I
```

So:

```text
A_k = q_k [[-alpha_k, -omega_k],
           [ omega_k, -alpha_k]]
alpha_k = softplus(raw_alpha_k) + eps
q_k = softplus(raw_q_k) + eps
```

The block energy is:

```text
E_k(x_k) = 0.5 x_k^T Q_k x_k
```

`J_k` is skew-symmetric and energy-preserving; `R_k` dissipates energy. The
implementation remains blockwise and real-valued.

## Synthetic Frequency Task

A minimal binary frequency classification task is available:

```bash
python run_train.py --dataset synthetic_frequency-classification --ssm_param resonant_2x2 --epochs 10 --jax_seed 0
```

Defaults:

```text
seq_len: 256
noise_std: 0.1
low_freq_range: [1, 3]
high_freq_range: [8, 12]
num_train: 1000
num_val: 200
num_test: 200
```

## Result Table Template

| Model | A parameterization | B,C real? | D enabled? | Hardware-friendly? | Accuracy | Loss | Params |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original S5 + D | original | maybe internal complex | yes | no | | | |
| Original S5 no D | original_no_D | maybe internal complex | no | partial | | | |
| Real decay | real_decay | yes | no | yes | | | |
| Resonant 2x2 | resonant_2x2 | yes | no | yes | | | |
| Energy-shaped 2x2 | energy_shaped_2x2 | yes | no | yes | | | |

The main scientific comparison should use the no-D models.
