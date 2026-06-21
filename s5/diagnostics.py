from pathlib import Path

import jax.numpy as jnp
import numpy as onp

from .ssm import discretize_bilinear, discretize_zoh
from .ssm_parameterizations import (
    SSM_PARAM_ENERGY_SHAPED_2X2,
    SSM_PARAM_REAL_DECAY,
    SSM_PARAM_RESONANT_2X2,
    discretize_2x2_blocks,
    discretize_real_decay,
    positive,
)


def _collect_ssm_modules(tree, modules):
    if not hasattr(tree, "items"):
        return
    if "B" in tree and ("C" in tree or "C1" in tree or "C2" in tree):
        modules.append(tree)
    for value in tree.values():
        _collect_ssm_modules(value, modules)


def _block_diag(blocks):
    n_blocks = blocks.shape[0]
    out = onp.zeros((2 * n_blocks, 2 * n_blocks), dtype=blocks.dtype)
    for i, block in enumerate(blocks):
        out[2 * i:2 * i + 2, 2 * i:2 * i + 2] = block
    return out


def _discrete_systems(params, args):
    modules = []
    _collect_ssm_modules(params, modules)
    systems = []

    for idx, module in enumerate(modules):
        if "raw_alpha" in module:
            alpha = positive(module["raw_alpha"])
            step = jnp.exp(module["log_step"][:, 0])

            if args.ssm_param == SSM_PARAM_REAL_DECAY:
                Lambda_bar, B_bar = discretize_real_decay(
                    alpha, module["B"], step, args.discretization
                )
                A = onp.diag(onp.asarray(Lambda_bar))
                B = onp.asarray(B_bar)
                C = onp.asarray(module["C"])
                eigs = onp.asarray(Lambda_bar)
            else:
                q = positive(module["raw_q"]) if args.ssm_param == SSM_PARAM_ENERGY_SHAPED_2X2 else jnp.ones_like(alpha)
                B_blocks = module["B"].reshape((module["B"].shape[0] // 2, 2, module["B"].shape[1]))
                A_bar, B_bar = discretize_2x2_blocks(
                    q * alpha, q * module["omega"], B_blocks, step, args.discretization
                )
                A = _block_diag(onp.asarray(A_bar))
                B = onp.asarray(B_bar).reshape((module["B"].shape[0], module["B"].shape[1]))
                C = onp.asarray(module["C"])
                eigs = onp.linalg.eigvals(A)

            systems.append({"layer": idx, "A": A, "B": B, "C": C, "eigs": eigs})
            continue

        Lambda = module["Lambda_re"] + 1j * module["Lambda_im"]
        if args.clip_eigs:
            Lambda = jnp.clip(module["Lambda_re"], None, -1e-4) + 1j * module["Lambda_im"]
        B_tilde = module["B"][..., 0] + 1j * module["B"][..., 1]
        step = jnp.exp(module["log_step"][:, 0])
        if args.discretization == "zoh":
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        else:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)

        if "C" in module:
            C_param = module["C"]
        else:
            C_param = module["C1"]
        C_tilde = C_param[..., 0] + 1j * C_param[..., 1]

        A = onp.diag(onp.asarray(Lambda_bar))
        systems.append({
            "layer": idx,
            "A": A,
            "B": onp.asarray(B_bar),
            "C": onp.asarray(C_tilde),
            "eigs": onp.asarray(Lambda_bar),
        })

    return systems


def _frequency_response(system, num_points):
    A, B, C = system["A"], system["B"], system["C"]
    freqs = onp.linspace(0.0, onp.pi, num_points)
    eye = onp.eye(A.shape[0], dtype=A.dtype)
    response = []
    for omega in freqs:
        z = onp.exp(1j * omega)
        H = C @ onp.linalg.solve(z * eye - A, B)
        response.append(onp.linalg.norm(H, ord="fro"))
    response = onp.asarray(response)
    response = response / (response.max() + 1e-12)
    return freqs / onp.pi, response


def plot_ssm_diagnostics(params, args):
    """Save learned SSM eigenvalue and frequency-response diagnostics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(getattr(args, "fig_dir", "./figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    systems = _discrete_systems(params, args)
    if not systems:
        print("[!] No SSM modules found; skipping diagnostic plots.")
        return []

    prefix = f"{args.dataset}_{args.ssm_param}_seed{args.jax_seed}"

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    theta = onp.linspace(0, 2 * onp.pi, 512)
    ax.plot(onp.cos(theta), onp.sin(theta), color="0.75", linewidth=1.0, label="unit circle")
    for system in systems:
        eigs = system["eigs"]
        ax.scatter(eigs.real, eigs.imag, s=14, alpha=0.75, label=f"layer {system['layer']}")
    ax.axhline(0, color="0.9", linewidth=0.8)
    ax.axvline(0, color="0.9", linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(eigenvalue of discrete A)")
    ax.set_ylabel("Im(eigenvalue of discrete A)")
    ax.set_title(f"{args.ssm_param} learned discrete A eigenvalues")
    ax.legend(fontsize=8)
    eig_path = out_dir / f"{prefix}_eigs.png"
    fig.tight_layout()
    fig.savefig(eig_path, dpi=getattr(args, "fig_dpi", 200))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for system in systems:
        freqs, response = _frequency_response(system, getattr(args, "freq_response_points", 256))
        ax.plot(freqs, response, label=f"layer {system['layer']}")
    ax.set_xlabel("normalized frequency (x pi rad/sample)")
    ax.set_ylabel("normalized ||H(e^{jw})||_F")
    ax.set_title(f"{args.ssm_param} SSM frequency response (D excluded)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fr_path = out_dir / f"{prefix}_freq_response.png"
    fig.tight_layout()
    fig.savefig(fr_path, dpi=getattr(args, "fig_dpi", 200))
    plt.close(fig)

    print(f"[*] Saved SSM diagnostic plots: {eig_path}, {fr_path}")
    return [str(eig_path), str(fr_path)]
