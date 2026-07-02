"""Digital-only hardware variation sweep.

This evaluates checkpoints after conductance projection and static component
variation, without running LTSpice. For each variation setting it writes:

* projected params: ``<out-dir>/<case>/params.msgpack``
* projected full-model netlist: ``<out-dir>/<case>/full_model.cir``
* projection report: ``<out-dir>/<case>/projection_report.json``
* aggregate summary: ``<out-dir>/summary.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

from jax import random

from run_train import normalize_config_overrides
from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel, RetrievalModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.ssm_parameterizations import (
    SSM_PARAM_CHOICES,
    SSM_PARAM_ORIGINAL,
    SSM_PARAM_ORIGINAL_NO_D,
    effective_use_D,
    init_RealValuedSSM,
    is_hardware_friendly,
)
from s5.train_helpers import create_train_state, validate
from s5.utils.util import str2bool

from .export_full_model import export_full_model
from .export_netlist import load_flax_params
from .hardware_projection import HardwareProjectionConfig
from .hardware_projector import save_projected_params


def _parse_sweep_values(values, cast):
    if values is None:
        return []
    parsed = []
    for value in values:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        for part in text.split(","):
            part = part.strip()
            if part:
                parsed.append(cast(part))
    return parsed


def _sigma_label(value):
    return "{:.6g}".format(float(value)).replace("-", "m").replace(".", "p")


def _case_name(sigma, variation_seed):
    return f"v{_sigma_label(sigma)}_seed{int(variation_seed)}"


def _create_dataset(args):
    create_dataset_fn = Datasets[args.dataset]
    if args.dataset in ["synthetic_frequency-classification"]:
        return create_dataset_fn(
            args.dir_name,
            seed=args.jax_seed,
            bsz=args.bsz,
            seq_len=args.synthetic_seq_len,
            noise_std=args.synthetic_noise_std,
            low_freq_range=args.synthetic_low_freq_range,
            high_freq_range=args.synthetic_high_freq_range,
            amplitude_range=args.synthetic_amplitude_range,
            bias_range=args.synthetic_bias_range,
            trend_range=args.synthetic_trend_range,
            distractor_count=args.synthetic_distractor_count,
            distractor_freq_range=args.synthetic_distractor_freq_range,
            distractor_amp_range=args.synthetic_distractor_amp_range,
            num_train=args.synthetic_num_train,
            num_val=args.synthetic_num_val,
            num_test=args.synthetic_num_test,
        )
    if args.dataset.startswith("ucr-"):
        return create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz, split_mode=args.ucr_split_mode)
    return create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)


def _build_model_and_state(args, n_classes, seq_len, in_dim, padded, retrieval):
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    use_D = effective_use_D(args.ssm_param, args.use_D)

    if is_hardware_friendly(args.ssm_param):
        ssm_init_fn = init_RealValuedSSM(
            H=args.d_model,
            P=ssm_size,
            ssm_param=args.ssm_param,
            discretization=args.discretization,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            bidirectional=args.bidirectional,
        )
    else:
        if args.ssm_param not in [SSM_PARAM_ORIGINAL, SSM_PARAM_ORIGINAL_NO_D]:
            raise ValueError(f"Unknown ssm_param {args.ssm_param}")
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        if args.conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2
        ssm_init_fn = init_S5SSM(
            H=args.d_model,
            P=ssm_size,
            Lambda_re_init=Lambda.real,
            Lambda_im_init=Lambda.imag,
            V=V,
            Vinv=V.conj().T,
            C_init=args.C_init,
            discretization=args.discretization,
            dt_min=args.dt_min,
            dt_max=args.dt_max,
            conj_sym=args.conj_sym,
            clip_eigs=args.clip_eigs,
            bidirectional=args.bidirectional,
            use_D=use_D,
        )

    if retrieval:
        model_cls = partial(
            RetrievalModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            layernorm=args.layernorm,
            bn_momentum=args.bn_momentum,
            use_residual=args.use_residual,
        )
    else:
        model_cls = partial(
            BatchClassificationModel,
            ssm=ssm_init_fn,
            d_output=n_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            padded=padded,
            activation=args.activation_fn,
            dropout=args.p_dropout,
            mode=args.mode,
            prenorm=args.prenorm,
            batchnorm=args.batchnorm,
            layernorm=args.layernorm,
            bn_momentum=args.bn_momentum,
            use_residual=args.use_residual,
        )

    state = create_train_state(
        model_cls,
        random.PRNGKey(args.jax_seed),
        padded,
        retrieval,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=args.ssm_lr_base,
        lr=args.lr_factor * args.ssm_lr_base,
        dt_global=args.dt_global,
    )
    return model_cls, state


def _projection_config(args, sigma, variation_seed):
    return HardwareProjectionConfig(
        hardware_projection="conductance",
        projection_scope=args.projection_scope,
        g_min=args.g_min,
        g_max=args.g_max,
        c_min=args.c_min,
        c_max=args.c_max,
        variation_sigma=float(sigma),
        variation_seed=int(variation_seed),
    )


def run(args):
    if args.batchnorm:
        raise ValueError("digital_variation_test currently expects checkpoints without batch_stats; use --batchnorm False.")
    if not is_hardware_friendly(args.ssm_param):
        raise ValueError("Hardware projection requires a hardware-friendly ssm_param.")

    _trainloader, _valloader, testloader, _aux, n_classes, seq_len, in_dim, _train_size = _create_dataset(args)
    padded = args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]
    retrieval = args.dataset in ["aan-classification"]
    model_cls, base_state = _build_model_and_state(args, n_classes, seq_len, in_dim, padded, retrieval)

    original_params = load_flax_params(args.params)
    original_state = base_state.replace(params=original_params)
    original_loss, original_acc = validate(
        original_state,
        model_cls,
        testloader,
        seq_len,
        in_dim,
        args.batchnorm,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sigmas = _parse_sweep_values(args.variation_sigma, float) or [0.0]
    seeds = _parse_sweep_values(args.variation_seed, int) or [0]

    rows = []
    for sigma in sigmas:
        for variation_seed in seeds:
            case = _case_name(sigma, variation_seed)
            case_dir = out_dir / case
            case_dir.mkdir(parents=True, exist_ok=True)
            config = _projection_config(args, sigma, variation_seed)

            projected_params_path, projection_report = save_projected_params(
                args.params,
                args.ssm_param,
                args.sample_rate,
                config,
                case_dir / "params.msgpack",
            )
            export_full_model(
                args.params,
                args.ssm_param,
                args.sample_rate,
                case_dir / "full_model.cir",
                json_out=case_dir / "full_model_manifest.json",
                projection_config=config,
                projection_report=case_dir / "projection_report.json",
            )

            projected_params = load_flax_params(projected_params_path)
            projected_state = base_state.replace(params=projected_params)
            test_loss, test_acc = validate(
                projected_state,
                model_cls,
                testloader,
                seq_len,
                in_dim,
                args.batchnorm,
            )
            aggregate = projection_report.get("aggregate", {})
            row = {
                "case": case,
                "variation_sigma": float(sigma),
                "variation_seed": int(variation_seed),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "clip_fraction": float(aggregate.get("clip_fraction", 0.0)),
                "num_clipped_low": int(aggregate.get("num_clipped_low", 0)),
                "num_clipped_high": int(aggregate.get("num_clipped_high", 0)),
                "num_conductances": int(aggregate.get("num_conductances", 0)),
                "projected_params": str(projected_params_path),
                "full_model_cir": str(case_dir / "full_model.cir"),
                "projection_report": str(case_dir / "projection_report.json"),
            }
            rows.append(row)
            print(
                "{case}: sigma={sigma:.6g} seed={seed} acc={acc:.4f} clip={clip:.4f}".format(
                    case=case,
                    sigma=float(sigma),
                    seed=int(variation_seed),
                    acc=float(test_acc),
                    clip=row["clip_fraction"],
                )
            )

    summary = {
        "params": str(args.params),
        "ssm_param": args.ssm_param,
        "sample_rate": float(args.sample_rate),
        "original": {
            "test_loss": float(original_loss),
            "test_accuracy": float(original_acc),
        },
        "projection": {
            "projection_scope": args.projection_scope,
            "g_min": float(args.g_min),
            "g_max": float(args.g_max),
            "c_min": float(args.c_min),
            "c_max": float(args.c_max),
        },
        "runs": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"original acc={float(original_acc):.4f}")
    print(f"wrote {summary_path}")
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True)
    parser.add_argument("--out-dir", default="out/digital_variation")
    parser.add_argument("--dataset", choices=Datasets.keys(), default="mnist-classification")
    parser.add_argument("--dir-name", "--dir_name", default="./cache_dir")
    parser.add_argument("--ucr-split-mode", "--ucr_split_mode", default="standard", choices=["standard", "combined"])

    parser.add_argument("--ssm-param", "--ssm_param", choices=SSM_PARAM_CHOICES, default="resonant_2x2")
    parser.add_argument("--sample-rate", "--sample_rate", type=float, default=160000.0)
    parser.add_argument("--n-layers", "--n_layers", type=int, default=2)
    parser.add_argument("--d-model", "--d_model", type=int, default=16)
    parser.add_argument("--ssm-size-base", "--ssm_size_base", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--C-init", "--C_init", default="trunc_standard_normal", choices=["trunc_standard_normal", "lecun_normal", "complex_normal"])
    parser.add_argument("--discretization", default="zoh", choices=["zoh", "bilinear"])
    parser.add_argument("--mode", default="last", choices=["pool", "last"])
    parser.add_argument("--use-D", "--use_D", type=str2bool, default=True)
    parser.add_argument("--use-residual", "--use_residual", type=str2bool, default=False)
    parser.add_argument("--activation-fn", "--activation_fn", default="relu", choices=["full_glu", "half_glu1", "half_glu2", "gelu", "relu"])
    parser.add_argument("--conj-sym", "--conj_sym", type=str2bool, default=True)
    parser.add_argument("--clip-eigs", "--clip_eigs", type=str2bool, default=False)
    parser.add_argument("--bidirectional", type=str2bool, default=False)
    parser.add_argument("--dt-min", "--dt_min", type=float, default=0.001)
    parser.add_argument("--dt-max", "--dt_max", type=float, default=0.1)

    parser.add_argument("--prenorm", type=str2bool, default=True)
    parser.add_argument("--batchnorm", type=str2bool, default=False)
    parser.add_argument("--layernorm", type=str2bool, default=False)
    parser.add_argument("--bn-momentum", "--bn_momentum", type=float, default=0.95)
    parser.add_argument("--p-dropout", "--p_dropout", type=float, default=0.0)
    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.05)
    parser.add_argument("--opt-config", "--opt_config", default="standard", choices=["standard", "BandCdecay", "BfastandCdecay", "noBCdecay"])
    parser.add_argument("--ssm-lr-base", "--ssm_lr_base", type=float, default=1e-3)
    parser.add_argument("--lr-factor", "--lr_factor", type=float, default=1.0)
    parser.add_argument("--dt-global", "--dt_global", type=str2bool, default=False)
    parser.add_argument("--jax-seed", "--jax_seed", type=int, default=1919)

    parser.add_argument("--projection-scope", "--projection_scope", default="block", choices=sorted({"global", "layer", "block", "row"}))
    parser.add_argument("--g-min", "--g_min", type=float, default=1e-6)
    parser.add_argument("--g-max", "--g_max", type=float, default=150e-6)
    parser.add_argument("--c-min", "--c_min", type=float, default=1e-12)
    parser.add_argument("--c-max", "--c_max", type=float, default=1e-9)
    parser.add_argument("--variation-sigma", "--variation_sigma", nargs="*", default=["0.0"])
    parser.add_argument("--variation-seed", "--variation_seed", nargs="*", default=["0"])

    parser.add_argument("--synthetic-seq-len", "--synthetic_seq_len", type=int, default=256)
    parser.add_argument("--synthetic-noise-std", "--synthetic_noise_std", type=float, default=0.25)
    parser.add_argument("--synthetic-low-freq-range", "--synthetic_low_freq_range", type=float, nargs=2, default=[4.0, 6.0])
    parser.add_argument("--synthetic-high-freq-range", "--synthetic_high_freq_range", type=float, nargs=2, default=[7.0, 9.0])
    parser.add_argument("--synthetic-amplitude-range", "--synthetic_amplitude_range", type=float, nargs=2, default=[0.5, 1.5])
    parser.add_argument("--synthetic-bias-range", "--synthetic_bias_range", type=float, nargs=2, default=[-0.3, 0.3])
    parser.add_argument("--synthetic-trend-range", "--synthetic_trend_range", type=float, nargs=2, default=[-0.3, 0.3])
    parser.add_argument("--synthetic-distractor-count", "--synthetic_distractor_count", type=int, default=1)
    parser.add_argument("--synthetic-distractor-freq-range", "--synthetic_distractor_freq_range", type=float, nargs=2, default=[1.0, 12.0])
    parser.add_argument("--synthetic-distractor-amp-range", "--synthetic_distractor_amp_range", type=float, nargs=2, default=[0.0, 0.4])
    parser.add_argument("--synthetic-num-train", "--synthetic_num_train", type=int, default=1000)
    parser.add_argument("--synthetic-num-val", "--synthetic_num_val", type=int, default=200)
    parser.add_argument("--synthetic-num-test", "--synthetic_num_test", type=int, default=200)
    raw_argv = sys.argv[1:] if argv is None else argv
    return parser.parse_args(normalize_config_overrides(raw_argv))


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
