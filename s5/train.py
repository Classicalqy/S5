from functools import partial
from pathlib import Path
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
from flax.serialization import to_bytes
import wandb

from .train_helpers import (
    calibration_epoch,
    create_hw_calibration_optimizer,
    create_train_state,
    linear_warmup,
    cosine_annealing,
    constant_lr,
    physical_noise_cvar_calibration_epoch,
    physical_noise_cvar_train_epoch,
    reduce_lr_on_plateau,
    reset_optimizer,
    stack_variation_offsets,
    train_epoch,
    validate,
    variation_aware_calibration_epoch,
)
from .dataloading import Datasets
from .seq_model import BatchClassificationModel, RetrievalModel
from .ssm import init_S5SSM
from .ssm_init import make_DPLR_HiPPO
from .ssm_parameterizations import (
    SSM_PARAM_ORIGINAL,
    SSM_PARAM_ORIGINAL_NO_D,
    effective_use_D,
    init_RealValuedSSM,
    is_hardware_friendly,
    summarize_state_space,
)
from .diagnostics import plot_ssm_diagnostics


def save_params_msgpack(params, out_path):
    """Save a Flax params tree in the format consumed by spice.export_netlist."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(to_bytes({"params": params}))
    return out_path


def _hw_calibration_enabled(args):
    return bool(getattr(args, "hw_calibrate_readout", False)) and (
        int(getattr(args, "hw_calibrate_epochs", 0)) > 0
        or int(getattr(args, "hw_variation_aware_epochs", 0)) > 0
    )


def _hw_variation_aware_enabled(args):
    return int(getattr(args, "hw_variation_aware_epochs", 0)) > 0


def _hw_projection_config(args, variation_sigma=None, variation_seed=None):
    from spice.hardware_projection import HardwareProjectionConfig

    return HardwareProjectionConfig(
        hardware_projection="conductance",
        projection_scope="block",
        g_min=getattr(args, "hw_g_min", 1e-6),
        g_max=getattr(args, "hw_g_max", 150e-6),
        c_min=getattr(args, "hw_c_min", 1e-12),
        c_max=getattr(args, "hw_c_max", 1e-6),
        variation_sigma=getattr(args, "hw_variation_sigma", 0.0) if variation_sigma is None else variation_sigma,
        variation_seed=getattr(args, "hw_variation_seed", 0) if variation_seed is None else variation_seed,
    )


def _hw_calibrated_params_out(args):
    path = getattr(args, "hw_calibrated_params_out", None)
    if path:
        return Path(path)

    from spice.hardware_projector import default_projected_params_path

    return default_projected_params_path(args.params_out)


def _hw_variation_aware_params_out(args):
    path = getattr(args, "hw_variation_aware_params_out", None)
    if path:
        return Path(path)
    path = Path(args.params_out)
    return path.with_name(f"{path.stem}_variation_aware{path.suffix or '.msgpack'}")


def _hw_variation_aware_train_samples(args):
    return max(1, int(getattr(args, "hw_variation_aware_train_samples", 3)))


def _hw_variation_aware_eval_samples(args):
    return max(1, int(getattr(args, "hw_variation_aware_eval_samples", 3)))


def _hw_variation_aware_select_sigma(args):
    value = getattr(args, "hw_variation_aware_select_sigma", None)
    if value is not None:
        value = float(value)
        if value < 0.0:
            raise ValueError("Variation-aware selection sigma must be non-negative.")
        return value
    return float(_hw_variation_aware_sigma_schedule(args)[-1])


def _hw_variation_aware_select_samples(args):
    value = getattr(args, "hw_variation_aware_select_samples", None)
    if value is None:
        return _hw_variation_aware_eval_samples(args)
    return max(1, int(value))


def _parse_hw_variation_aware_sigma_schedule(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        pieces = []
        for item in value:
            pieces.extend(str(item).replace(",", " ").split())
    else:
        pieces = str(value).replace(",", " ").split()
    values = [float(piece) for piece in pieces if piece]
    if any(sigma < 0.0 for sigma in values):
        raise ValueError("Variation-aware sigma schedule values must be non-negative.")
    return values


def _hw_variation_aware_sigma_schedule(args):
    schedule = _parse_hw_variation_aware_sigma_schedule(
        getattr(args, "hw_variation_aware_sigma_schedule", None)
    )
    if schedule:
        return schedule
    return [float(getattr(args, "hw_variation_aware_sigma", 0.0))]


def _hw_variation_aware_epoch_sigma(args, epoch):
    schedule = _hw_variation_aware_sigma_schedule(args)
    return float(schedule[min(int(epoch), len(schedule) - 1)])


def _hw_train_noise_sigma_schedule(args):
    schedule = _parse_hw_variation_aware_sigma_schedule(
        getattr(args, "hw_train_noise_sigma_schedule", None)
    )
    if schedule:
        return schedule
    return [float(getattr(args, "hw_train_noise_sigma", 0.0))]


def _hw_train_noise_epoch_sigma(args, epoch):
    schedule = _hw_train_noise_sigma_schedule(args)
    return float(schedule[min(int(epoch), len(schedule) - 1)])


def _hw_train_noise_enabled(args):
    return any(sigma > 0.0 for sigma in _hw_train_noise_sigma_schedule(args))


def _hw_train_noise_samples(args):
    return max(1, int(getattr(args, "hw_train_noise_samples", 4)))


def _hw_variation_aware_nominal_fraction(args):
    value = float(getattr(args, "hw_variation_aware_nominal_fraction", 0.0))
    return min(1.0, max(0.0, value))


def _hw_variation_aware_nominal_train_samples(train_samples, nominal_fraction):
    train_samples = max(1, int(train_samples))
    nominal_fraction = min(1.0, max(0.0, float(nominal_fraction)))
    if nominal_fraction <= 0.0:
        return 0
    return min(train_samples, max(1, int(round(train_samples * nominal_fraction))))


def _hw_variation_aware_train_seed(base_seed, epoch, sample_index, train_samples):
    return int(base_seed) + int(epoch) * int(train_samples) + int(sample_index)


def _hw_variation_aware_eval_seed(base_seed, epoch, sample_index, eval_samples):
    return int(base_seed) + 10000 + int(epoch) * int(eval_samples) + int(sample_index)


def _hw_variation_aware_select_seed(base_seed, sample_index):
    """Return a held-out, epoch-invariant seed for checkpoint selection."""
    return int(base_seed) + 20000 + int(sample_index)


def _hw_variation_aware_score(val_accuracies, select_metric):
    values = [float(value) for value in val_accuracies]
    if not values:
        raise ValueError("Cannot score variation-aware checkpoint without validation accuracies.")
    if select_metric == "mean_acc":
        return float(np.mean(np.asarray(values, dtype=np.float32)))
    if select_metric == "mean_std":
        values = np.asarray(values, dtype=np.float32)
        return float(np.mean(values) - 0.5 * np.std(values))
    if select_metric == "mean_std_strong":
        values = np.asarray(values, dtype=np.float32)
        return float(np.mean(values) - np.std(values))
    if select_metric == "min_acc":
        return float(min(values))
    if select_metric == "p10_acc":
        values = sorted(values)
        if len(values) == 1:
            return float(values[0])
        position = 0.10 * (len(values) - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        weight = position - lower
        return float(values[lower] * (1.0 - weight) + values[upper] * weight)
    raise ValueError(f"Unknown variation-aware selection metric: {select_metric}")


def train(args):
    """
    Main function to train over a certain number of epochs
    """

    best_test_loss = 100000000
    best_test_acc = -10000.0

    if args.USE_WANDB:
        # Make wandb config dictionary
        wandb.init(project=args.wandb_project, job_type='model_training', config=vars(args), entity=args.wandb_entity)
    else:
        wandb.init(mode='offline')

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base
    use_D = effective_use_D(args.ssm_param, args.use_D)

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    print("[*] Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    if args.dataset in ["imdb-classification", "listops-classification", "aan-classification"]:
        padded = True
        if args.dataset in ["aan-classification"]:
            # Use retreival model for document matching
            retrieval = True
            print("Using retrieval model for document matching")
        else:
            retrieval = False

    else:
        padded = False
        retrieval = False

    # For speech dataset
    if args.dataset in ["speech10-classification", "speech35-classification"]:
        speech = True
        print("Will evaluate on both resolutions for speech task")
    else:
        speech = False

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    if args.dataset in ["synthetic_frequency-classification"]:
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
          create_dataset_fn(args.dir_name,
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
                            num_test=args.synthetic_num_test)
    elif args.dataset.startswith("ucr-"):
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
          create_dataset_fn(args.dir_name,
                            seed=args.jax_seed,
                            bsz=args.bsz,
                            split_mode=args.ucr_split_mode)
    else:
        trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
          create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)

    print(f"[*] Starting S5 Training on `{args.dataset}` =>> Initializing...")

    if is_hardware_friendly(args.ssm_param):
        if args.conj_sym:
            print("[!] conj_sym is ignored for real-valued hardware-friendly parameterizations.")
        ssm_init_fn = init_RealValuedSSM(H=args.d_model,
                                         P=ssm_size,
                                         ssm_param=args.ssm_param,
                                         discretization=args.discretization,
                                         dt_min=args.dt_min,
                                         dt_max=args.dt_max,
                                         bidirectional=args.bidirectional)
    else:
        if args.ssm_param not in [SSM_PARAM_ORIGINAL, SSM_PARAM_ORIGINAL_NO_D]:
            raise ValueError("Unknown ssm_param {}".format(args.ssm_param))

        # Initialize state matrix A using approximation to HiPPO-LegS matrix.
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

        if args.conj_sym:
            block_size = block_size // 2
            ssm_size = ssm_size // 2

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        # If initializing state matrix A as block-diagonal, put HiPPO approximation
        # on each block.
        Lambda = (Lambda * np.ones((args.blocks, block_size))).ravel()
        V = block_diag(*([V] * args.blocks))
        Vinv = block_diag(*([Vc] * args.blocks))

        print("Lambda.shape={}".format(Lambda.shape))
        print("V.shape={}".format(V.shape))
        print("Vinv.shape={}".format(Vinv.shape))

        ssm_init_fn = init_S5SSM(H=args.d_model,
                                 P=ssm_size,
                                 Lambda_re_init=Lambda.real,
                                 Lambda_im_init=Lambda.imag,
                                 V=V,
                                 Vinv=Vinv,
                                 C_init=args.C_init,
                                 discretization=args.discretization,
                                 dt_min=args.dt_min,
                                 dt_max=args.dt_max,
                                 conj_sym=args.conj_sym,
                                 clip_eigs=args.clip_eigs,
                                 bidirectional=args.bidirectional,
                                 use_D=use_D)

    if retrieval:
        # Use retrieval head for AAN task
        print("Using Retrieval head for {} task".format(args.dataset))
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

    # initialize training state
    state = create_train_state(model_cls,
                               init_rng,
                               padded,
                               retrieval,
                               in_dim=in_dim,
                               bsz=args.bsz,
                               seq_len=seq_len,
                               weight_decay=args.weight_decay,
                               batchnorm=args.batchnorm,
                               opt_config=args.opt_config,
                               ssm_lr=ssm_lr,
                               lr=lr,
                               dt_global=args.dt_global)

    ssm_summary = summarize_state_space(state.params, args, use_D)
    print("[*] SSM configuration summary:")
    for key, value in ssm_summary.items():
        print("    {}: {}".format(key, value))
    if is_hardware_friendly(args.ssm_param):
        if not ssm_summary["B_real"]:
            print("[!] Hardware-friendly run has complex B.")
        if not ssm_summary["C_real"]:
            print("[!] Hardware-friendly run has complex C.")
        if ssm_summary["D_enabled"]:
            print("[!] Hardware-friendly run has nonzero D enabled.")
        if not ssm_summary["A_real_block_equivalent"]:
            print("[!] Hardware-friendly run lacks a real-valued A interpretation.")
    if wandb.run is not None:
        wandb.run.summary.update(ssm_summary)
        wandb.log({k: v for k, v in ssm_summary.items() if isinstance(v, (int, float, bool))})

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = 100000000, -100000000.0, 0  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    best_params = state.params
    best_batch_stats = state.batch_stats if args.batchnorm and hasattr(state, "batch_stats") else None
    steps_per_epoch = int(train_size/args.bsz)
    train_noise_enabled = _hw_train_noise_enabled(args)
    train_noise_sigma_schedule = _hw_train_noise_sigma_schedule(args)
    train_noise_samples = _hw_train_noise_samples(args)
    train_noise_consistency_weight = float(getattr(args, "hw_train_noise_consistency_weight", 0.5))
    train_noise_cvar_fraction = float(getattr(args, "hw_train_noise_cvar_fraction", 0.5))
    if train_noise_enabled:
        if args.batchnorm:
            raise ValueError("physical-noise normal training requires batchnorm=False.")
        if not is_hardware_friendly(args.ssm_param):
            raise ValueError("physical-noise normal training requires a hardware-friendly ssm_param.")
        print(
            "[*] Enabled differentiable physical-noise normal training with sigma_schedule={}, "
            "samples={}, consistency_weight={:.6g}, cvar_fraction={:.6g}.".format(
                ",".join("{:.6g}".format(sigma) for sigma in train_noise_sigma_schedule),
                train_noise_samples,
                train_noise_consistency_weight,
                train_noise_cvar_fraction,
            )
        )
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch+1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch+1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (steps_per_epoch * args.warmup_end)
        else:
            print("using constant lr for epoch {}".format(epoch+1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (decay_function, ssm_lr, lr, step, end_step, args.opt_config, args.lr_min)

        train_rng, skey = random.split(train_rng)
        train_noise_sigma = _hw_train_noise_epoch_sigma(args, epoch)
        if train_noise_enabled:
            print(
                "[*] Training Epoch {} physical-noise sigma={:.6g}.".format(
                    epoch + 1, train_noise_sigma
                )
            )
            state, train_loss, step = physical_noise_cvar_train_epoch(
                state,
                skey,
                model_cls,
                trainloader,
                seq_len,
                in_dim,
                args.batchnorm,
                lr_params,
                train_noise_sigma,
                args.ssm_param,
                train_noise_samples,
                train_noise_consistency_weight,
                train_noise_cvar_fraction,
            )
        else:
            state, train_loss, step = train_epoch(state,
                                                  skey,
                                                  model_cls,
                                                  trainloader,
                                                  seq_len,
                                                  in_dim,
                                                  args.batchnorm,
                                                  lr_params)

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(state,
                                         model_cls,
                                         valloader,
                                         seq_len,
                                         in_dim,
                                         args.batchnorm)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(state,
                                           model_cls,
                                           testloader,
                                           seq_len,
                                           in_dim,
                                           args.batchnorm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # else use test set as validation set (e.g. IMDB)
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(state,
                                         model_cls,
                                         testloader,
                                         seq_len,
                                         in_dim,
                                         args.batchnorm)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            best_params = state.params
            if args.batchnorm and hasattr(state, "batch_stats"):
                best_batch_stats = state.batch_stats
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

            # Do some validation on improvement.
            if speech:
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_loss, val2_acc = validate(state,
                                               model_cls,
                                               aux_dataloaders['valloader2'],
                                               int(seq_len // 2),
                                               in_dim,
                                               args.batchnorm,
                                               step_rescale=2.0)

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_loss, test2_acc = validate(state, model_cls, aux_dataloaders['testloader2'], int(seq_len // 2), in_dim, args.batchnorm, step_rescale=2.0)
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss: {test2_loss:.5f} --"
                    f" Val Accuracy: {val2_acc:.4f}"
                    f" Test Accuracy: {test2_acc:.4f}"
                )

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(input, factor=args.reduce_factor, patience=args.lr_patience, lr_min=args.lr_min)

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )

        if valloader is not None:
            if speech:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "Val2 loss": val2_loss,
                        "Val2 Accuracy": val2_acc,
                        "Test2 Loss": test2_loss,
                        "Test2 Accuracy": test2_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )
            else:
                wandb.log(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        "count": count,
                        "Learning rate count": lr_count,
                        "Opt acc": opt_acc,
                        "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                        "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                    }
                )

        else:
            wandb.log(
                {
                    "Training Loss": train_loss,
                    "Val loss": val_loss,
                    "Val Accuracy": val_acc,
                    "count": count,
                    "Learning rate count": lr_count,
                    "Opt acc": opt_acc,
                    "lr": state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'],
                    "ssm_lr": state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate']
                }
            )
        wandb.run.summary["Best Val Loss"] = best_loss
        wandb.run.summary["Best Val Accuracy"] = best_acc
        wandb.run.summary["Best Epoch"] = best_epoch
        wandb.run.summary["Best Test Loss"] = best_test_loss
        wandb.run.summary["Best Test Accuracy"] = best_test_acc

        if count > args.early_stop_patience:
            break

    if getattr(args, "save_params", False) or _hw_calibration_enabled(args):
        out_path = save_params_msgpack(best_params, args.params_out)
        print("[*] Saved best normal-training params to {}".format(out_path))

    if _hw_calibration_enabled(args):
        if not is_hardware_friendly(args.ssm_param):
            raise ValueError("--hw_calibrate_readout requires a hardware-friendly ssm_param.")

        from spice.hardware_projector import project_params_tree

        hw_calibrate_mode = getattr(args, "hw_calibrate_mode", "readout")
        hw_project_each_epoch = bool(getattr(args, "hw_project_each_calibration_epoch", True))
        if hw_calibrate_mode not in {"readout", "analog"}:
            raise ValueError("--hw_calibrate_mode must be one of: readout, analog.")

        print(f"[*] Starting hardware {hw_calibrate_mode} calibration from best normal-training params...")
        projection_config = _hw_projection_config(args)
        projected_params, projection_report = project_params_tree(
            params=best_params,
            ssm_param=args.ssm_param,
            sample_rate=getattr(args, "hw_sample_rate", 16000.0),
            projection_config=projection_config,
        )
        aggregate = projection_report.get("aggregate", {})
        print(
            "[*] Hardware projection clip fraction: {:.6f} (low={}, high={}, total={})".format(
                aggregate.get("clip_fraction", 0.0),
                aggregate.get("num_clipped_low", 0),
                aggregate.get("num_clipped_high", 0),
                aggregate.get("num_conductances", 0),
            )
        )

        state = state.replace(params=projected_params)
        if args.batchnorm and best_batch_stats is not None and hasattr(state, "batch_stats"):
            state = state.replace(batch_stats=best_batch_stats)

        if valloader is not None:
            print(f"[*] Running projected hardware params Validation before {hw_calibrate_mode} calibration...")
            projected_val_loss, projected_val_acc = validate(
                state,
                model_cls,
                valloader,
                seq_len,
                in_dim,
                args.batchnorm,
            )
            print(f"[*] Running projected hardware params Test before {hw_calibrate_mode} calibration...")
            projected_test_loss, projected_test_acc = validate(
                state,
                model_cls,
                testloader,
                seq_len,
                in_dim,
                args.batchnorm,
            )
        else:
            print(f"[*] Running projected hardware params Test before {hw_calibrate_mode} calibration...")
            projected_val_loss, projected_val_acc = validate(
                state,
                model_cls,
                testloader,
                seq_len,
                in_dim,
                args.batchnorm,
            )
            projected_test_loss, projected_test_acc = projected_val_loss, projected_val_acc

        print(f"\n=>> Projected Hardware Params Metrics Before {hw_calibrate_mode.capitalize()} Calibration ===")
        print(
            f"\tVal Loss: {projected_val_loss:.5f} -- Test Loss: {projected_test_loss:.5f} --"
            f" Val Accuracy: {projected_val_acc:.4f}"
            f" Test Accuracy: {projected_test_acc:.4f}"
        )
        wandb.log(
            {
                "HW Projected Val loss": projected_val_loss,
                "HW Projected Val Accuracy": projected_val_acc,
                "HW Projected Test Loss": projected_test_loss,
                "HW Projected Test Accuracy": projected_test_acc,
            }
        )
        wandb.run.summary["HW Projected Val Loss"] = projected_val_loss
        wandb.run.summary["HW Projected Val Accuracy"] = projected_val_acc
        wandb.run.summary["HW Projected Test Loss"] = projected_test_loss
        wandb.run.summary["HW Projected Test Accuracy"] = projected_test_acc

        state = reset_optimizer(
            state,
            create_hw_calibration_optimizer(
                state.params,
                getattr(args, "hw_calibrate_lr", 1e-4),
                hw_calibrate_mode,
            ),
        )

        best_calibrated_params = state.params
        best_calibrated_batch_stats = state.batch_stats if args.batchnorm and hasattr(state, "batch_stats") else None
        best_calibrated_loss, best_calibrated_acc = 100000000, -100000000.0
        best_calibrated_test_loss, best_calibrated_test_acc = 100000000, -100000000.0
        best_calibrated_epoch = 0

        for cal_epoch in range(int(args.hw_calibrate_epochs)):
            print(f"[*] Starting HW Calibration Epoch {cal_epoch + 1}...")
            train_rng, skey = random.split(train_rng)
            state, cal_train_loss = calibration_epoch(
                state,
                skey,
                model_cls,
                trainloader,
                seq_len,
                in_dim,
                args.batchnorm,
            )

            if hw_calibrate_mode == "analog" and hw_project_each_epoch:
                projected_params, projection_report = project_params_tree(
                    params=state.params,
                    ssm_param=args.ssm_param,
                    sample_rate=getattr(args, "hw_sample_rate", 16000.0),
                    projection_config=projection_config,
                )
                state = state.replace(params=projected_params)
                aggregate = projection_report.get("aggregate", {})
                print(
                    "[*] HW analog epoch projection clip fraction: {:.6f} (low={}, high={}, total={})".format(
                        aggregate.get("clip_fraction", 0.0),
                        aggregate.get("num_clipped_low", 0),
                        aggregate.get("num_clipped_high", 0),
                        aggregate.get("num_conductances", 0),
                    )
                )

            if valloader is not None:
                print(f"[*] Running HW Calibration Epoch {cal_epoch + 1} Validation...")
                cal_val_loss, cal_val_acc = validate(
                    state,
                    model_cls,
                    valloader,
                    seq_len,
                    in_dim,
                    args.batchnorm,
                )
                print(f"[*] Running HW Calibration Epoch {cal_epoch + 1} Test...")
                cal_test_loss, cal_test_acc = validate(
                    state,
                    model_cls,
                    testloader,
                    seq_len,
                    in_dim,
                    args.batchnorm,
                )
            else:
                print(f"[*] Running HW Calibration Epoch {cal_epoch + 1} Test...")
                cal_val_loss, cal_val_acc = validate(
                    state,
                    model_cls,
                    testloader,
                    seq_len,
                    in_dim,
                    args.batchnorm,
                )
                cal_test_loss, cal_test_acc = cal_val_loss, cal_val_acc

            print(f"\n=>> HW Calibration Epoch {cal_epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {cal_train_loss:.5f} -- Val Loss: {cal_val_loss:.5f} --"
                f" Test Loss: {cal_test_loss:.5f} -- Val Accuracy: {cal_val_acc:.4f}"
                f" Test Accuracy: {cal_test_acc:.4f}"
            )

            if cal_val_acc > best_calibrated_acc:
                best_calibrated_loss = cal_val_loss
                best_calibrated_acc = cal_val_acc
                best_calibrated_test_loss = cal_test_loss
                best_calibrated_test_acc = cal_test_acc
                best_calibrated_epoch = cal_epoch
                best_calibrated_params = state.params
                if args.batchnorm and hasattr(state, "batch_stats"):
                    best_calibrated_batch_stats = state.batch_stats

            wandb.log(
                {
                    "HW Calibration Training Loss": cal_train_loss,
                    "HW Calibration Val loss": cal_val_loss,
                    "HW Calibration Val Accuracy": cal_val_acc,
                    "HW Calibration Test Loss": cal_test_loss,
                    "HW Calibration Test Accuracy": cal_test_acc,
                }
            )

        calibrated_out = save_params_msgpack(best_calibrated_params, _hw_calibrated_params_out(args))
        print("[*] Saved best nominal hardware-calibrated params to {}".format(calibrated_out))
        wandb.run.summary["HW Calibration Params Out"] = str(calibrated_out)

        if _hw_variation_aware_enabled(args):
            aware_sigma_schedule = _hw_variation_aware_sigma_schedule(args)
            aware_seed = int(getattr(args, "hw_variation_aware_seed", 0))
            aware_train_samples = _hw_variation_aware_train_samples(args)
            aware_select_sigma = _hw_variation_aware_select_sigma(args)
            aware_select_samples = _hw_variation_aware_select_samples(args)
            aware_select_metric = getattr(args, "hw_variation_aware_select_metric", "mean_acc")
            aware_nominal_gate = max(0.0, float(getattr(args, "hw_variation_aware_nominal_gate", 0.0)))
            aware_loss = getattr(args, "hw_variation_aware_loss", "projected_eot")
            aware_consistency_weight = float(getattr(args, "hw_variation_aware_consistency_weight", 0.5))
            aware_cvar_fraction = float(getattr(args, "hw_variation_aware_cvar_fraction", 0.5))
            aware_nominal_fraction = _hw_variation_aware_nominal_fraction(args)
            aware_nominal_samples = _hw_variation_aware_nominal_train_samples(
                aware_train_samples,
                aware_nominal_fraction,
            )
            if aware_loss == "physical_noise_cvar" and args.batchnorm:
                raise ValueError("physical_noise_cvar variation-aware training requires batchnorm=False.")
            nominal_config = _hw_projection_config(args, variation_sigma=0.0)
            print(
                "[*] Starting variation-aware analog calibration with sigma_schedule={}, "
                "loss={}, train_samples={}, nominal_train_samples={}, select_sigma={}, select_samples={}, "
                "select_metric={}, nominal_gate={:.6g}...".format(
                    ",".join("{:.6g}".format(sigma) for sigma in aware_sigma_schedule),
                    aware_loss,
                    aware_train_samples,
                    aware_nominal_samples,
                    aware_select_sigma,
                    aware_select_samples,
                    aware_select_metric,
                    aware_nominal_gate,
                )
            )

            nominal_params, projection_report = project_params_tree(
                params=best_calibrated_params,
                ssm_param=args.ssm_param,
                sample_rate=getattr(args, "hw_sample_rate", 16000.0),
                projection_config=nominal_config,
            )
            state = state.replace(params=nominal_params)
            state = reset_optimizer(
                state,
                create_hw_calibration_optimizer(
                    state.params,
                    getattr(args, "hw_calibrate_lr", 1e-4),
                    "analog",
                ),
            )

            best_aware_master_params = state.params
            best_aware_batch_stats = state.batch_stats if args.batchnorm and hasattr(state, "batch_stats") else None
            best_aware_loss, best_aware_acc = best_calibrated_loss, best_calibrated_acc
            best_aware_test_loss, best_aware_test_acc = best_calibrated_test_loss, best_calibrated_test_acc
            best_aware_epoch = 0
            best_aware_selected = False

            for aware_epoch in range(int(args.hw_variation_aware_epochs)):
                aware_sigma = _hw_variation_aware_epoch_sigma(args, aware_epoch)
                train_rng, skey = random.split(train_rng)
                if aware_loss == "projected_eot":
                    train_configs = []
                    for sample_index in range(aware_train_samples):
                        sample_sigma = 0.0 if sample_index < aware_nominal_samples else aware_sigma
                        train_configs.append(
                            _hw_projection_config(
                                args,
                                variation_sigma=sample_sigma,
                                variation_seed=_hw_variation_aware_train_seed(
                                    aware_seed,
                                    aware_epoch,
                                    sample_index,
                                    aware_train_samples,
                                ),
                            )
                        )

                    varied_params = [
                        project_params_tree(
                            params=state.params,
                            ssm_param=args.ssm_param,
                            sample_rate=getattr(args, "hw_sample_rate", 16000.0),
                            projection_config=config,
                        )[0]
                        for config in train_configs
                    ]
                    variation_offsets = stack_variation_offsets(state.params, varied_params)

                    print(
                        "[*] Starting HW Variation-Aware Epoch {} EOT calibration "
                        "(train sigma={:.6g}, {} fixed realizations)...".format(
                            aware_epoch + 1,
                            aware_sigma,
                            aware_train_samples,
                        )
                    )
                    state, aware_train_loss = variation_aware_calibration_epoch(
                        state,
                        skey,
                        model_cls,
                        trainloader,
                        seq_len,
                        in_dim,
                        args.batchnorm,
                        variation_offsets,
                        aware_train_samples,
                    )
                elif aware_loss == "physical_noise_cvar":
                    print(
                        "[*] Starting HW Variation-Aware Epoch {} physical-noise CVaR calibration "
                        "(train sigma={:.6g}, samples={}, consistency_weight={:.6g}, cvar_fraction={:.6g})...".format(
                            aware_epoch + 1,
                            aware_sigma,
                            aware_train_samples,
                            aware_consistency_weight,
                            aware_cvar_fraction,
                        )
                    )
                    state, aware_train_loss = physical_noise_cvar_calibration_epoch(
                        state,
                        skey,
                        model_cls,
                        trainloader,
                        seq_len,
                        in_dim,
                        args.batchnorm,
                        aware_sigma,
                        args.ssm_param,
                        aware_train_samples,
                        aware_consistency_weight,
                        aware_cvar_fraction,
                    )
                else:
                    raise ValueError("--hw_variation_aware_loss must be projected_eot or physical_noise_cvar.")
                aware_train_loss = float(aware_train_loss)
                val_losses = []
                val_accs = []
                test_losses = []
                test_accs = []
                for sample_index in range(aware_select_samples):
                    variation_seed = _hw_variation_aware_select_seed(aware_seed, sample_index)
                    varied_config = _hw_projection_config(
                        args,
                        variation_sigma=aware_select_sigma,
                        variation_seed=variation_seed,
                    )
                    eval_params, projection_report = project_params_tree(
                        params=state.params,
                        ssm_param=args.ssm_param,
                        sample_rate=getattr(args, "hw_sample_rate", 16000.0),
                        projection_config=varied_config,
                    )
                    eval_state = state.replace(params=eval_params)
                    if valloader is not None:
                        print(
                            "[*] Running HW Variation-Aware Epoch {} Selection Validation Sample {} (sigma={:.6g}, seed={})...".format(
                                aware_epoch + 1,
                                sample_index + 1,
                                aware_select_sigma,
                                variation_seed,
                            )
                        )
                        aware_val_loss, aware_val_acc = validate(
                            eval_state,
                            model_cls,
                            valloader,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                        )
                        print(
                            "[*] Running HW Variation-Aware Epoch {} Selection Test Sample {} (sigma={:.6g}, seed={})...".format(
                                aware_epoch + 1,
                                sample_index + 1,
                                aware_select_sigma,
                                variation_seed,
                            )
                        )
                        aware_test_loss, aware_test_acc = validate(
                            eval_state,
                            model_cls,
                            testloader,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                        )
                    else:
                        print(
                            "[*] Running HW Variation-Aware Epoch {} Selection Test Sample {} (sigma={:.6g}, seed={})...".format(
                                aware_epoch + 1,
                                sample_index + 1,
                                aware_select_sigma,
                                variation_seed,
                            )
                        )
                        aware_val_loss, aware_val_acc = validate(
                            eval_state,
                            model_cls,
                            testloader,
                            seq_len,
                            in_dim,
                            args.batchnorm,
                        )
                        aware_test_loss, aware_test_acc = aware_val_loss, aware_val_acc
                    val_losses.append(float(aware_val_loss))
                    val_accs.append(float(aware_val_acc))
                    test_losses.append(float(aware_test_loss))
                    test_accs.append(float(aware_test_acc))

                aware_val_loss = float(np.mean(np.asarray(val_losses, dtype=np.float32)))
                aware_val_acc = _hw_variation_aware_score(val_accs, aware_select_metric)
                aware_test_loss = float(np.mean(np.asarray(test_losses, dtype=np.float32)))
                aware_test_acc = float(np.mean(np.asarray(test_accs, dtype=np.float32)))
                nominal_eval_params, _nominal_projection_report = project_params_tree(
                    params=state.params,
                    ssm_param=args.ssm_param,
                    sample_rate=getattr(args, "hw_sample_rate", 16000.0),
                    projection_config=nominal_config,
                )
                nominal_eval_state = state.replace(params=nominal_eval_params)
                if valloader is not None:
                    print(f"[*] Running HW Variation-Aware Epoch {aware_epoch + 1} Nominal Validation...")
                    aware_nominal_val_loss, aware_nominal_val_acc = validate(
                        nominal_eval_state,
                        model_cls,
                        valloader,
                        seq_len,
                        in_dim,
                        args.batchnorm,
                    )
                    print(f"[*] Running HW Variation-Aware Epoch {aware_epoch + 1} Nominal Test...")
                    aware_nominal_test_loss, aware_nominal_test_acc = validate(
                        nominal_eval_state,
                        model_cls,
                        testloader,
                        seq_len,
                        in_dim,
                        args.batchnorm,
                    )
                else:
                    print(f"[*] Running HW Variation-Aware Epoch {aware_epoch + 1} Nominal Test...")
                    aware_nominal_val_loss, aware_nominal_val_acc = validate(
                        nominal_eval_state,
                        model_cls,
                        testloader,
                        seq_len,
                        in_dim,
                        args.batchnorm,
                    )
                    aware_nominal_test_loss, aware_nominal_test_acc = aware_nominal_val_loss, aware_nominal_val_acc
                aware_nominal_val_acc = float(aware_nominal_val_acc)
                aware_nominal_test_acc = float(aware_nominal_test_acc)
                nominal_gate_pass = aware_nominal_gate <= 0.0 or aware_nominal_val_acc >= aware_nominal_gate
                aggregate = projection_report.get("aggregate", {})
                print(f"\n=>> HW Variation-Aware Epoch {aware_epoch + 1} Metrics ===")
                print(
                    f"\tTrain Loss: {aware_train_loss:.5f} -- Mean Var Val Loss: {aware_val_loss:.5f} --"
                    f" Mean Var Test Loss: {aware_test_loss:.5f} -- Mean Var Val Accuracy: {aware_val_acc:.4f}"
                    f" Mean Var Test Accuracy: {aware_test_acc:.4f} -- Train Sigma: {aware_sigma:.6g}"
                    f" -- Selection Sigma: {aware_select_sigma:.6g}"
                    f" -- Nominal Val Accuracy: {aware_nominal_val_acc:.4f}"
                    f" -- Nominal Gate Pass: {nominal_gate_pass}"
                )

                if nominal_gate_pass and (not best_aware_selected or aware_val_acc > best_aware_acc):
                    best_aware_loss = aware_val_loss
                    best_aware_acc = aware_val_acc
                    best_aware_test_loss = aware_test_loss
                    best_aware_test_acc = aware_test_acc
                    best_aware_epoch = aware_epoch
                    best_aware_master_params = state.params
                    best_aware_selected = True
                    if args.batchnorm and hasattr(state, "batch_stats"):
                        best_aware_batch_stats = state.batch_stats

                wandb.log(
                    {
                        "HW Variation-Aware Training Loss": aware_train_loss,
                        "HW Variation-Aware Val loss": aware_val_loss,
                        "HW Variation-Aware Val Accuracy": aware_val_acc,
                        "HW Variation-Aware Test Loss": aware_test_loss,
                        "HW Variation-Aware Test Accuracy": aware_test_acc,
                        "HW Variation-Aware Train Sigma": aware_sigma,
                        "HW Variation-Aware Selection Sigma": aware_select_sigma,
                        "HW Variation-Aware Nominal Val Accuracy": aware_nominal_val_acc,
                        "HW Variation-Aware Nominal Test Accuracy": aware_nominal_test_acc,
                        "HW Variation-Aware Nominal Gate Pass": float(nominal_gate_pass),
                    }
                )

            if aware_nominal_gate > 0.0 and not best_aware_selected:
                print(
                    "[!] No variation-aware checkpoint passed nominal_gate={:.6g}; "
                    "keeping the best nominal calibrated params.".format(aware_nominal_gate)
                )

            best_aware_params, projection_report = project_params_tree(
                params=best_aware_master_params,
                ssm_param=args.ssm_param,
                sample_rate=getattr(args, "hw_sample_rate", 16000.0),
                projection_config=nominal_config,
            )
            best_calibrated_params = best_aware_params
            best_calibrated_batch_stats = best_aware_batch_stats
            best_calibrated_loss = best_aware_loss
            best_calibrated_acc = best_aware_acc
            best_calibrated_test_loss = best_aware_test_loss
            best_calibrated_test_acc = best_aware_test_acc
            best_calibrated_epoch = best_aware_epoch
            wandb.run.summary["HW Variation-Aware Best Val Loss"] = best_aware_loss
            wandb.run.summary["HW Variation-Aware Best Val Accuracy"] = best_aware_acc
            wandb.run.summary["HW Variation-Aware Best Epoch"] = best_aware_epoch
            wandb.run.summary["HW Variation-Aware Best Test Loss"] = best_aware_test_loss
            wandb.run.summary["HW Variation-Aware Best Test Accuracy"] = best_aware_test_acc
            aware_out = save_params_msgpack(best_aware_params, _hw_variation_aware_params_out(args))
            print("[*] Saved best variation-aware hardware-calibrated params to {}".format(aware_out))
            wandb.run.summary["HW Variation-Aware Params Out"] = str(aware_out)

        state = state.replace(params=best_calibrated_params)
        if args.batchnorm and best_calibrated_batch_stats is not None and hasattr(state, "batch_stats"):
            state = state.replace(batch_stats=best_calibrated_batch_stats)

        wandb.run.summary["HW Calibration Best Val Loss"] = best_calibrated_loss
        wandb.run.summary["HW Calibration Best Val Accuracy"] = best_calibrated_acc
        wandb.run.summary["HW Calibration Best Epoch"] = best_calibrated_epoch
        wandb.run.summary["HW Calibration Best Test Loss"] = best_calibrated_test_loss
        wandb.run.summary["HW Calibration Best Test Accuracy"] = best_calibrated_test_acc
        wandb.run.summary["HW Projection Clip Fraction"] = aggregate.get("clip_fraction", 0.0)

    if getattr(args, "plot_ssm_diagnostics", False):
        plot_ssm_diagnostics(state.params, args)
