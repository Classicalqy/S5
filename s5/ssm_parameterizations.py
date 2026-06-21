from functools import partial

import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn import softplus
from jax.nn.initializers import lecun_normal

from .ssm_init import init_log_steps


SSM_PARAM_ORIGINAL = "original"
SSM_PARAM_ORIGINAL_NO_D = "original_no_D"
SSM_PARAM_REAL_DECAY = "real_decay"
SSM_PARAM_RESONANT_2X2 = "resonant_2x2"
SSM_PARAM_ENERGY_SHAPED_2X2 = "energy_shaped_2x2"

SSM_PARAM_CHOICES = (
    SSM_PARAM_ORIGINAL,
    SSM_PARAM_ORIGINAL_NO_D,
    SSM_PARAM_REAL_DECAY,
    SSM_PARAM_RESONANT_2X2,
    SSM_PARAM_ENERGY_SHAPED_2X2,
)

HARDWARE_FRIENDLY_PARAMS = (
    SSM_PARAM_REAL_DECAY,
    SSM_PARAM_RESONANT_2X2,
    SSM_PARAM_ENERGY_SHAPED_2X2,
)

POSITIVE_EPS = 1e-4


def is_hardware_friendly(ssm_param):
    return ssm_param in HARDWARE_FRIENDLY_PARAMS


def effective_use_D(ssm_param, requested_use_D):
    if ssm_param == SSM_PARAM_ORIGINAL:
        return requested_use_D
    if requested_use_D:
        print("[!] D was requested but is disabled for {}.".format(ssm_param))
    return False


def inverse_softplus(x):
    return np.log(np.expm1(x))


def positive(raw, eps=POSITIVE_EPS):
    return softplus(raw) + eps


def binary_operator_block(q_i, q_j):
    """Parallel-scan composition for independent real 2x2 state blocks."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return (
        np.einsum("...kij,...kjl->...kil", A_j, A_i),
        np.einsum("...kij,...kj->...ki", A_j, b_i) + b_j,
    )


def apply_real_diagonal_ssm(Lambda_bar, B_bar, C, input_sequence, bidirectional):
    Lambda_elements = Lambda_bar * np.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)
    _, xs = jax.lax.associative_scan(
        lambda qi, qj: (qj[0] * qi[0], qj[0] * qi[1] + qj[1]),
        (Lambda_elements, Bu_elements),
    )

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(
            lambda qi, qj: (qj[0] * qi[0], qj[0] * qi[1] + qj[1]),
            (Lambda_elements, Bu_elements),
            reverse=True,
        )
        xs = np.concatenate((xs, xs2), axis=-1)

    return jax.vmap(lambda x: C @ x)(xs)


def apply_real_block_ssm(A_bar, B_bar, C, input_sequence, bidirectional):
    A_elements = A_bar * np.ones((input_sequence.shape[0],) + A_bar.shape)
    Bu_elements = jax.vmap(lambda u: np.einsum("kdh,h->kd", B_bar, u))(input_sequence)
    _, xs = jax.lax.associative_scan(binary_operator_block, (A_elements, Bu_elements))

    if bidirectional:
        _, xs2 = jax.lax.associative_scan(
            binary_operator_block,
            (A_elements, Bu_elements),
            reverse=True,
        )
        xs = np.concatenate((xs.reshape((xs.shape[0], -1)), xs2.reshape((xs2.shape[0], -1))), axis=-1)
    else:
        xs = xs.reshape((xs.shape[0], -1))

    return jax.vmap(lambda x: C @ x)(xs)


def make_2x2_blocks(a, w):
    row0 = np.stack((-a, -w), axis=-1)
    row1 = np.stack((w, -a), axis=-1)
    return np.stack((row0, row1), axis=-2)


def invert_2x2(M):
    a, b = M[:, 0, 0], M[:, 0, 1]
    c, d = M[:, 1, 0], M[:, 1, 1]
    det = a * d - b * c
    row0 = np.stack((d / det, -b / det), axis=-1)
    row1 = np.stack((-c / det, a / det), axis=-1)
    return np.stack((row0, row1), axis=-2)


def discretize_real_decay(alpha, B, step, discretization):
    if discretization == "zoh":
        Lambda_bar = np.exp(-alpha * step)
        B_bar = ((1.0 - Lambda_bar) / alpha)[:, None] * B
    elif discretization == "bilinear":
        denom = 1.0 + 0.5 * step * alpha
        Lambda_bar = (1.0 - 0.5 * step * alpha) / denom
        B_bar = (step / denom)[:, None] * B
    else:
        raise NotImplementedError("Discretization method {} not implemented".format(discretization))
    return Lambda_bar, B_bar


def discretize_2x2_blocks(alpha, omega, B_blocks, step, discretization):
    A = make_2x2_blocks(alpha, omega)
    I = np.broadcast_to(np.eye(2), A.shape)

    if discretization == "zoh":
        decay = np.exp(-alpha * step)
        angle = omega * step
        c, s = np.cos(angle), np.sin(angle)
        A_bar = decay[:, None, None] * np.stack(
            (
                np.stack((c, -s), axis=-1),
                np.stack((s, c), axis=-1),
            ),
            axis=-2,
        )
        # Exact zero-order-hold B discretization: integral exp(A tau) B dtau.
        B_bar = np.einsum("kij,kjl,klh->kih", invert_2x2(A), A_bar - I, B_blocks)
    elif discretization == "bilinear":
        left = I - 0.5 * step[:, None, None] * A
        right = I + 0.5 * step[:, None, None] * A
        left_inv = invert_2x2(left)
        A_bar = np.einsum("kij,kjl->kil", left_inv, right)
        B_bar = np.einsum("kij,kjh->kih", left_inv, step[:, None, None] * B_blocks)
    else:
        raise NotImplementedError("Discretization method {} not implemented".format(discretization))

    return A_bar, B_bar


def init_log_spaced_positive(min_value, max_value, count):
    values = np.exp(np.linspace(np.log(min_value), np.log(max_value), count))
    return inverse_softplus(values - POSITIVE_EPS)


class RealValuedSSM(nn.Module):
    """Hardware-friendly real-valued SSM parameterizations.

    The resonant and energy-shaped variants are implemented as explicit real
    2x2 blocks rather than unconstrained complex B/C systems.  Their dynamics
    are equivalent to conjugate-pair eigenvalues, but B, C, and the scan state
    are all real-valued.
    """

    H: int
    P: int
    ssm_param: str
    discretization: str
    dt_min: float
    dt_max: float
    bidirectional: bool = False
    use_D: bool = False
    step_rescale: float = 1.0

    def setup(self):
        if self.ssm_param not in HARDWARE_FRIENDLY_PARAMS:
            raise ValueError("{} is not a hardware-friendly parameterization.".format(self.ssm_param))
        if self.use_D:
            raise ValueError("D must be disabled for {}.".format(self.ssm_param))
        if self.ssm_param in [SSM_PARAM_RESONANT_2X2, SSM_PARAM_ENERGY_SHAPED_2X2] and self.P % 2 != 0:
            raise ValueError("resonant_2x2 and energy_shaped_2x2 require an even state dimension.")

        output_state_dim = self.P * (2 if self.bidirectional else 1)
        self.B = self.param("B", lecun_normal(), (self.P, self.H))
        self.C = self.param("C", lecun_normal(), (self.H, output_state_dim))

        if self.ssm_param == SSM_PARAM_REAL_DECAY:
            self.raw_alpha = self.param(
                "raw_alpha",
                lambda rng, shape: init_log_spaced_positive(0.1, 10.0, shape[0]),
                (self.P,),
            )
            self.log_step = self.param("log_step", init_log_steps, (self.P, self.dt_min, self.dt_max))
            step = self.step_rescale * np.exp(self.log_step[:, 0])
            self.alpha = positive(self.raw_alpha)
            self.Lambda_bar, self.B_bar = discretize_real_decay(
                self.alpha, self.B, step, self.discretization
            )
        else:
            n_blocks = self.P // 2
            self.raw_alpha = self.param(
                "raw_alpha",
                lambda rng, shape: init_log_spaced_positive(0.1, 3.0, shape[0]),
                (n_blocks,),
            )
            omega_max = 0.8 * np.pi / self.dt_max
            self.omega = self.param(
                "omega",
                lambda rng, shape: np.linspace(0.1, omega_max, shape[0]),
                (n_blocks,),
            )
            if self.ssm_param == SSM_PARAM_ENERGY_SHAPED_2X2:
                self.raw_q = self.param(
                    "raw_q",
                    lambda rng, shape: inverse_softplus(np.ones(shape) - POSITIVE_EPS),
                    (n_blocks,),
                )
                q = positive(self.raw_q)
            else:
                q = np.ones((n_blocks,))

            self.log_step = self.param("log_step", init_log_steps, (n_blocks, self.dt_min, self.dt_max))
            step = self.step_rescale * np.exp(self.log_step[:, 0])
            self.alpha = positive(self.raw_alpha)
            self.q = q
            B_blocks = self.B.reshape((n_blocks, 2, self.H))
            self.A_bar, self.B_bar = discretize_2x2_blocks(
                q * self.alpha, q * self.omega, B_blocks, step, self.discretization
            )

        self.D = np.zeros((self.H,))

    def __call__(self, input_sequence):
        if self.ssm_param == SSM_PARAM_REAL_DECAY:
            return apply_real_diagonal_ssm(
                self.Lambda_bar, self.B_bar, self.C, input_sequence, self.bidirectional
            )

        return apply_real_block_ssm(
            self.A_bar, self.B_bar, self.C, input_sequence, self.bidirectional
        )


def init_RealValuedSSM(
    H,
    P,
    ssm_param,
    discretization,
    dt_min,
    dt_max,
    bidirectional=False,
):
    return partial(
        RealValuedSSM,
        H=H,
        P=P,
        ssm_param=ssm_param,
        discretization=discretization,
        dt_min=dt_min,
        dt_max=dt_max,
        bidirectional=bidirectional,
        use_D=False,
    )


def is_complex_array(x):
    return x.dtype in [np.complex64, np.complex128]


def count_parameters(params):
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size * (2 if is_complex_array(x) else 1) for x in leaves))


def count_ssm_parameters(params):
    ssm_names = {
        "B", "C", "C1", "C2", "D", "Lambda_re", "Lambda_im", "log_step",
        "norm", "raw_alpha", "omega", "raw_q",
    }

    def walk(tree, parent_key=None):
        total = 0
        for key, value in tree.items():
            if hasattr(value, "items"):
                total += walk(value, key)
            elif key in ssm_names:
                total += value.size * (2 if is_complex_array(value) else 1)
        return total

    return int(walk(params))


def _collect_ssm_modules(tree, modules):
    if not hasattr(tree, "items"):
        return
    if "B" in tree and ("C" in tree or "C1" in tree or "C2" in tree):
        modules.append(tree)
    for value in tree.values():
        _collect_ssm_modules(value, modules)


def _stats(values):
    if not values:
        return {}
    values = np.concatenate([np.ravel(v) for v in values])
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def summarize_state_space(params, args, use_D):
    modules = []
    _collect_ssm_modules(params, modules)
    total = count_parameters(params)
    ssm_total = count_ssm_parameters(params)

    summary = {
        "ssm_param": args.ssm_param,
        "use_D": bool(use_D),
        "B_real": True,
        "C_real": True,
        "D_enabled": bool(use_D),
        "A_real_block_equivalent": args.ssm_param in [SSM_PARAM_ORIGINAL_NO_D] + list(HARDWARE_FRIENDLY_PARAMS),
        "state_dim": int(args.ssm_size_base),
        "model_dim": int(args.d_model),
        "num_layers": int(args.n_layers),
        "total_parameters": total,
        "SSM_parameters": ssm_total,
        "non_SSM_parameters": total - ssm_total,
    }

    spectral = []
    alphas, omegas, qs = [], [], []
    for module in modules:
        if "B" in module:
            summary["B_real"] = summary["B_real"] and not is_complex_array(module["B"])
        if "C" in module:
            summary["C_real"] = summary["C_real"] and not is_complex_array(module["C"])
        if "C1" in module:
            summary["C_real"] = summary["C_real"] and not is_complex_array(module["C1"])
        if "C2" in module:
            summary["C_real"] = summary["C_real"] and not is_complex_array(module["C2"])

        if "raw_alpha" in module:
            alpha = positive(module["raw_alpha"])
            step = np.exp(module["log_step"][:, 0])
            if "raw_q" in module:
                q = positive(module["raw_q"])
                qs.append(q)
                spectral.append(np.exp(-q * alpha * step))
            elif "omega" in module:
                spectral.append(np.exp(-alpha * step))
            else:
                spectral.append(np.exp(-alpha * step))
            alphas.append(alpha)
            if "omega" in module:
                omegas.append(module["omega"])
        elif "Lambda_re" in module:
            Lambda = module["Lambda_re"] + 1j * module["Lambda_im"]
            if args.clip_eigs:
                Lambda = np.clip(module["Lambda_re"], None, -1e-4) + 1j * module["Lambda_im"]
            step = np.exp(module["log_step"][:, 0])
            if args.discretization == "zoh":
                spectral.append(np.abs(np.exp(Lambda * step)))
            else:
                spectral.append(np.abs((1 + 0.5 * step * Lambda) / (1 - 0.5 * step * Lambda)))

    if spectral:
        summary["spectral_radius_discrete_A"] = float(np.max(np.concatenate([np.ravel(v) for v in spectral])))

    alpha_stats = _stats(alphas)
    if alpha_stats:
        summary.update({f"alpha_{k}": v for k, v in alpha_stats.items()})

    omega_stats = _stats(omegas)
    if omega_stats:
        summary.update({f"omega_{k}": v for k, v in omega_stats.items()})

    q_stats = _stats(qs)
    if q_stats:
        summary.update({f"q_{k}": v for k, v in q_stats.items()})

    return summary
