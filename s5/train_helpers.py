from functools import partial
import math
import jax
import jax.numpy as np
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any, Tuple

from .ssm_parameterizations import HARDWARE_FRIENDLY_PARAMS, POSITIVE_EPS


SSM_PARAMETER_NAMES = [
    "B", "Lambda_re", "Lambda_im", "log_step", "norm",
    "raw_alpha", "omega", "raw_q",
]
SSM_WITH_C_PARAMETER_NAMES = SSM_PARAMETER_NAMES + ["C", "C1", "C2", "D"]
ANALOG_CALIBRATION_SSM_PARAMETER_NAMES = {
    "B", "C", "raw_alpha", "omega", "raw_q", "log_step",
}
PHYSICAL_NOISE_DIRECT_PARAMETER_NAMES = {"B", "C", "omega"}
PHYSICAL_NOISE_POSITIVE_PARAMETER_NAMES = {"raw_alpha", "raw_q"}


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step,  lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states['regular'].inner_state.hyperparams['learning_rate'] = np.array(lr_val, dtype=np.float32)
    state.opt_state.inner_states['ssm'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states['none'].inner_state.hyperparams['learning_rate'] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: (map_fn(v) if hasattr(v, "keys") else fn(k, v))
            for k, v in nested_dict.items()
        }

    return map_fn


def map_nested_with_path_fn(fn):
    """Recursively apply `fn(path, key, value)` to leaves in a nested dict."""

    def map_fn(nested_dict, path=()):
        return {
            k: (
                map_fn(v, path + (k,))
                if hasattr(v, "keys")
                else fn(path + (k,), k, v)
            )
            for k, v in nested_dict.items()
        }

    return map_fn


def maybe_unfreeze(tree):
    """Return a mutable parameter tree across Flax versions.

    Older Flax returns FrozenDicts with an .unfreeze() method, while newer
    versions may already return ordinary dicts from model.init().
    """
    return tree.unfreeze() if hasattr(tree, "unfreeze") else tree


def positive_to_raw(value):
    value = np.maximum(value - POSITIVE_EPS, np.finfo(np.float32).tiny)
    return np.where(value > 20.0, value, np.log(np.expm1(value)))


def signed_multiplicative_noise(value, sigma, rng):
    if float(sigma) <= 0.0:
        return value
    return value * (1.0 + float(sigma) * jax.random.normal(rng, value.shape, dtype=value.dtype))


def _positive_multiplicative_noise(raw_value, sigma, rng):
    physical_value = jax.nn.softplus(raw_value) + POSITIVE_EPS
    noisy_value = signed_multiplicative_noise(physical_value, sigma, rng)
    noisy_value = np.maximum(noisy_value, POSITIVE_EPS + np.finfo(noisy_value.dtype).tiny)
    return positive_to_raw(noisy_value).astype(raw_value.dtype)


def _stable_path_hash(path):
    value = 0
    for part in path:
        for char in part:
            value = (value * 131 + ord(char)) & 0x7FFFFFFF
        value = (value * 131 + 17) & 0x7FFFFFFF
    return value


def perturb_physical_params(params, rng, sigma, ssm_param):
    """Apply differentiable physical-parameter noise to hardware-friendly SSM leaves."""
    if ssm_param not in HARDWARE_FRIENDLY_PARAMS:
        raise ValueError("physical-noise training requires a hardware-friendly ssm_param.")
    if float(sigma) <= 0.0:
        return params

    def perturb(path, key, value):
        if key not in PHYSICAL_NOISE_DIRECT_PARAMETER_NAMES | PHYSICAL_NOISE_POSITIVE_PARAMETER_NAMES:
            return value
        leaf_rng = jax.random.fold_in(rng, _stable_path_hash(path))
        if key in PHYSICAL_NOISE_DIRECT_PARAMETER_NAMES:
            return signed_multiplicative_noise(value, sigma, leaf_rng).astype(value.dtype)
        return _positive_multiplicative_noise(value, sigma, leaf_rng)

    return map_nested_with_path_fn(perturb)(params)


def decoder_only_param_labels(params):
    """Label only top-level decoder params as trainable."""
    return map_nested_with_path_fn(
        lambda path, _key, _value: "decoder" if path and path[0] == "decoder" else "frozen"
    )(params)


def analog_calibration_param_labels(params):
    """Label hardware-realizable analog params as trainable."""

    def label(path, key, _value):
        if path and path[0] == "decoder":
            return "analog"
        if path[:2] == ("encoder", "encoder") or (
            path and path[0] == "encoder" and len(path) == 2
        ):
            return "analog"
        if key in ANALOG_CALIBRATION_SSM_PARAMETER_NAMES:
            return "analog"
        return "frozen"

    return map_nested_with_path_fn(label)(params)


def create_decoder_only_optimizer(params, learning_rate):
    labels = decoder_only_param_labels(params)
    return optax.multi_transform(
        {
            "decoder": optax.adam(learning_rate=learning_rate),
            "frozen": optax.set_to_zero(),
        },
        labels,
    )


def create_analog_calibration_optimizer(params, learning_rate):
    labels = analog_calibration_param_labels(params)
    return optax.multi_transform(
        {
            "analog": optax.adam(learning_rate=learning_rate),
            "frozen": optax.set_to_zero(),
        },
        labels,
    )


def create_hw_calibration_optimizer(params, learning_rate, mode):
    if mode == "readout":
        return create_decoder_only_optimizer(params, learning_rate)
    if mode == "analog":
        return create_analog_calibration_optimizer(params, learning_rate)
    raise ValueError(f"Unknown hardware calibration mode: {mode}")


def reset_optimizer(state, tx):
    return state.replace(step=0, tx=tx, opt_state=tx.init(state.params))


def create_train_state(model_cls,
                       rng,
                       padded,
                       retrieval,
                       in_dim=1,
                       bsz=128,
                       seq_len=784,
                       weight_decay=0.01,
                       batchnorm=False,
                       opt_config="standard",
                       ssm_lr=1e-3,
                       lr=1e-3,
                       dt_global=False
                       ):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param dt_global:
    :return:
    """

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (np.ones((2*bsz, seq_len, in_dim)), np.ones(2*bsz))
            integration_timesteps = np.ones((2*bsz, seq_len,))
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones((bsz, seq_len,))
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))
        integration_timesteps = np.ones((bsz, seq_len, ))

    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng,
                            "dropout": dropout_rng},
                           dummy_input, integration_timesteps,
                           )
    if batchnorm:
        params = maybe_unfreeze(variables["params"])
        batch_stats = variables["batch_stats"]
    else:
        params = maybe_unfreeze(variables["params"])
        # Note: `unfreeze()` is for using Optax.

    if opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
            SSM parameters with no weight decay.
        """
        print("configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in [p for p in SSM_PARAMETER_NAMES if p != "log_step"]
                else ("none" if k in [] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in SSM_PARAMETER_NAMES
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
           ssm learning rate to B.
        """
        print("configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in [p for p in SSM_PARAMETER_NAMES if p not in ["B", "log_step"]]
                else ("none" if k in ["B"] else "regular")
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in [p for p in SSM_PARAMETER_NAMES if p != "B"]
                else ("none" if k in ["B"] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=ssm_lr,
                                                              weight_decay=weight_decay),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply 
           faster global learning rate to B also.
        """
        print("configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in [p for p in SSM_PARAMETER_NAMES if p not in ["B", "log_step"]]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in [p for p in SSM_PARAMETER_NAMES if p != "B"]
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.adamw)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included 
            with the SSM parameters and uses ssm learning rate.
         """
        print("configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in [p for p in SSM_WITH_C_PARAMETER_NAMES if p != "log_step"]
                else ("none" if k in [] else "regular")
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: "ssm"
                if k in SSM_WITH_C_PARAMETER_NAMES
                else ("none" if k in [] else "regular")
            )
        tx = optax.multi_transform(
            {
                "none": optax.inject_hyperparams(optax.sgd)(learning_rate=0.0),
                "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
                "regular": optax.inject_hyperparams(optax.adamw)(learning_rate=lr,
                                                                 weight_decay=weight_decay),
            },
            ssm_fn,
        )

    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(params)
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    if batchnorm:
        class TrainState(train_state.TrainState):
            batch_stats: Any
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def prep_batch(batch: tuple,
               seq_len: int,
               in_dim: int) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    inputs = np.asarray(inputs.numpy())

    # Grab lengths from aux if it is there.
    lengths = aux_data.get('lengths', None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(inputs, ((0, 0), (0, num_pad)), 'constant', constant_values=(0,))

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    # Convert and apply.
    targets = np.array(targets.numpy())

    # If there is an aux channel containing the integration times, then add that.
    if 'timesteps' in aux_data.keys():
        integration_timesteps = np.diff(np.asarray(aux_data['timesteps'].numpy()))
    else:
        integration_timesteps = np.ones((len(inputs), seq_len))

    return full_inputs, targets.astype(float), integration_timesteps


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm, lr_params):
    """
    Training function for an epoch that loops over batches.
    """
    # Store Metrics
    model = model(training=True)
    batch_losses = []

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
        )
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, np.mean(np.array(batch_losses)), step


def physical_noise_cvar_train_epoch(
    state,
    rng,
    model,
    trainloader,
    seq_len,
    in_dim,
    batchnorm,
    lr_params,
    sigma,
    ssm_param,
    num_samples,
    consistency_weight,
    cvar_fraction,
    ema_params=None,
    mesa_weight=0.0,
    mesa_beta=0.999,
):
    """Run one normal-training epoch with differentiable physical noise."""
    if batchnorm:
        raise ValueError("physical-noise training requires batchnorm=False.")
    model = model(training=True)
    batch_losses = []
    num_samples = max(1, int(num_samples))

    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    for batch in tqdm(trainloader):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        rng, sample_rng = jax.random.split(rng)
        sample_rngs = jax.random.split(sample_rng, num_samples)
        state, loss = physical_noise_cvar_train_step(
            state,
            sample_rngs,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            sigma,
            ssm_param,
            consistency_weight,
            cvar_fraction,
            ema_params=ema_params,
            mesa_weight=mesa_weight,
        )
        batch_losses.append(loss)
        lr_params = (decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)
        if ema_params is not None:
            ema_params = update_ema_params(ema_params, state.params, mesa_beta)

    return state, np.mean(np.asarray(batch_losses)), step, ema_params


def calibration_epoch(state, rng, model, trainloader, seq_len, in_dim, batchnorm):
    """Run one fixed-LR calibration epoch without touching scheduler state."""
    model = model(training=True)
    batch_losses = []

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(
            state,
            drop_rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
        )
        batch_losses.append(loss)

    return state, np.mean(np.array(batch_losses))


def variation_aware_calibration_epoch(
    state,
    rng,
    model,
    trainloader,
    seq_len,
    in_dim,
    batchnorm,
    variation_offsets,
    num_samples,
):
    """Calibrate master params using the mean gradient from fixed chip samples.

    ``variation_offsets`` stacks one projected-minus-master offset per hardware
    realization along axis zero. The offsets stay fixed for this epoch, while
    the master tree is updated after every batch.
    """
    model = model(training=True)
    batch_losses = []
    num_samples = max(1, int(num_samples))

    for batch in tqdm(trainloader):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        rng, sample_rng = jax.random.split(rng)
        sample_rngs = jax.random.split(sample_rng, num_samples)
        state, loss = variation_aware_train_step(
            state,
            sample_rngs,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            variation_offsets,
        )
        batch_losses.append(loss)

    return state, np.mean(np.asarray(batch_losses))


def cvar_top_mean(losses, cvar_fraction):
    losses = np.asarray(losses)
    num_losses = int(losses.shape[0])
    fraction = min(1.0, max(0.0, float(cvar_fraction)))
    k = max(1, min(num_losses, int(math.ceil(num_losses * fraction))))
    top_losses, _ = jax.lax.top_k(losses, k)
    return np.mean(top_losses)


def update_ema_params(ema_params, params, beta):
    beta = float(beta)
    return jax.tree_util.tree_map(
        lambda ema, current: beta * ema + (1.0 - beta) * current,
        ema_params,
        params,
    )


def physical_noise_cvar_train_step(
    state,
    rngs,
    batch_inputs,
    batch_labels,
    batch_integration_timesteps,
    model,
    batchnorm,
    sigma,
    ssm_param,
    consistency_weight,
    cvar_fraction,
    ema_params=None,
    mesa_weight=0.0,
):
    if batchnorm:
        raise ValueError("physical_noise_cvar variation-aware training requires batchnorm=False.")
    mesa_weight = float(mesa_weight)
    if float(sigma) <= 0.0 and mesa_weight <= 0.0:
        rng = np.asarray(rngs)[0]
        return train_step(
            state,
            rng,
            batch_inputs,
            batch_labels,
            batch_integration_timesteps,
            model,
            batchnorm,
        )
    if ema_params is None:
        ema_params = state.params
    return _physical_noise_cvar_train_step(
        state,
        ema_params,
        rngs,
        batch_inputs,
        batch_labels,
        batch_integration_timesteps,
        model,
        batchnorm,
        float(sigma),
        ssm_param,
        float(consistency_weight),
        float(cvar_fraction),
        mesa_weight,
    )


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11, 12))
def _physical_noise_cvar_train_step(
    state,
    ema_params,
    rngs,
    batch_inputs,
    batch_labels,
    batch_integration_timesteps,
    model,
    batchnorm,
    sigma,
    ssm_param,
    consistency_weight,
    cvar_fraction,
    mesa_weight,
):
    rngs = np.asarray(rngs)

    def apply_params(params, rng):
        logits, _mod_vars = model.apply(
            {"params": params},
            batch_inputs,
            batch_integration_timesteps,
            rngs={"dropout": rng},
            mutable=["intermediates"],
        )
        return logits

    def loss_fn(params):
        nominal_logits = apply_params(params, rngs[0])
        nominal_losses = cross_entropy_loss(nominal_logits, batch_labels)
        nominal_ce = np.mean(nominal_losses)
        teacher_logits = jax.lax.stop_gradient(nominal_logits)
        teacher_probs = jax.lax.stop_gradient(np.exp(nominal_logits))

        def noisy_loss(noise_rng):
            noisy_params = perturb_physical_params(params, noise_rng, sigma, ssm_param)
            noisy_logits = apply_params(noisy_params, noise_rng)
            noisy_ce = np.mean(cross_entropy_loss(noisy_logits, batch_labels))
            kl = np.mean(np.sum(teacher_probs * (teacher_logits - noisy_logits), axis=-1))
            return noisy_ce, kl

        noisy_losses, consistency_losses = jax.vmap(noisy_loss)(rngs)
        noisy_cvar_ce = cvar_top_mean(noisy_losses, cvar_fraction)
        consistency_kl = np.mean(consistency_losses)
        if mesa_weight > 0.0:
            ema_logits = jax.lax.stop_gradient(apply_params(ema_params, rngs[0]))
            ema_probs = jax.lax.stop_gradient(np.exp(ema_logits))
            mesa_kl = np.mean(np.sum(ema_probs * (ema_logits - nominal_logits), axis=-1))
        else:
            mesa_kl = 0.0
        loss = 0.5 * nominal_ce + 0.5 * noisy_cvar_ce + consistency_weight * consistency_kl
        loss = loss + mesa_weight * mesa_kl
        return loss, (nominal_ce, noisy_cvar_ce, consistency_kl, mesa_kl)

    (loss, _metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def physical_noise_cvar_calibration_epoch(
    state,
    rng,
    model,
    trainloader,
    seq_len,
    in_dim,
    batchnorm,
    sigma,
    ssm_param,
    num_samples,
    consistency_weight,
    cvar_fraction,
):
    if batchnorm:
        raise ValueError("physical_noise_cvar variation-aware training requires batchnorm=False.")
    model = model(training=True)
    batch_losses = []
    num_samples = max(1, int(num_samples))

    for batch in tqdm(trainloader):
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        rng, sample_rng = jax.random.split(rng)
        sample_rngs = jax.random.split(sample_rng, num_samples)
        state, loss = physical_noise_cvar_train_step(
            state,
            sample_rngs,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            sigma,
            ssm_param,
            consistency_weight,
            cvar_fraction,
        )
        batch_losses.append(loss)

    return state, np.mean(np.asarray(batch_losses))


def validate(state, model, testloader, seq_len, in_dim, batchnorm, step_rescale=1.0):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)
        loss, acc, pred = eval_step(inputs, labels, integration_timesteps, state, model, batchnorm)
        losses = np.append(losses, loss)
        accuracies = np.append(accuracies, acc)

    aveloss, aveaccu = np.mean(losses), np.mean(accuracies)
    return aveloss, aveaccu


@partial(jax.jit, static_argnums=(5, 6))
def train_step(state,
               rng,
               batch_inputs,
               batch_labels,
               batch_integration_timesteps,
               model,
               batchnorm,
               ):
    """Performs a single training step given a batch of data"""
    def loss_fn(params):

        if batchnorm:
            logits, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates", "batch_stats"],
            )
        else:
            logits, mod_vars = model.apply(
                {"params": params},
                batch_inputs, batch_integration_timesteps,
                rngs={"dropout": rng},
                mutable=["intermediates"],
            )

        loss = np.mean(cross_entropy_loss(logits, batch_labels))

        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


def stack_variation_offsets(master_params, varied_params):
    """Stack projected-minus-master offsets for batched EOT training."""
    if not varied_params:
        raise ValueError("variation-aware training requires at least one realization.")
    return jax.tree_util.tree_map(
        lambda master, *varied: np.stack(
            [np.asarray(params) - np.asarray(master) for params in varied], axis=0
        ),
        master_params,
        *varied_params,
    )


@partial(jax.jit, static_argnums=(5, 6))
def variation_aware_train_step(
    state,
    rngs,
    batch_inputs,
    batch_labels,
    batch_integration_timesteps,
    model,
    batchnorm,
    variation_offsets,
):
    """Run a GPU-batched EOT/straight-through master-parameter update."""
    rngs = np.asarray(rngs)

    def loss_and_grads_for_realization(offset, rng):
        def loss_fn(master_params):
            varied_params = jax.tree_util.tree_map(
                lambda master, delta: master + jax.lax.stop_gradient(delta),
                master_params,
                offset,
            )
            if batchnorm:
                logits, mod_vars = model.apply(
                    {"params": varied_params, "batch_stats": state.batch_stats},
                    batch_inputs,
                    batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates", "batch_stats"],
                )
            else:
                logits, mod_vars = model.apply(
                    {"params": varied_params},
                    batch_inputs,
                    batch_integration_timesteps,
                    rngs={"dropout": rng},
                    mutable=["intermediates"],
                )
            return np.mean(cross_entropy_loss(logits, batch_labels)), mod_vars

        return jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    (losses, mod_vars), grads = jax.vmap(loss_and_grads_for_realization)(
        variation_offsets, rngs
    )
    mean_grads = jax.tree_util.tree_map(lambda leaves: np.mean(leaves, axis=0), grads)
    mean_loss = np.mean(losses)
    if batchnorm:
        mean_batch_stats = jax.tree_util.tree_map(
            lambda leaves: np.mean(leaves, axis=0), mod_vars["batch_stats"]
        )
        state = state.apply_gradients(grads=mean_grads, batch_stats=mean_batch_stats)
    else:
        state = state.apply_gradients(grads=mean_grads)
    return state, mean_loss


@partial(jax.jit, static_argnums=(4, 5))
def eval_step(batch_inputs,
              batch_labels,
              batch_integration_timesteps,
              state,
              model,
              batchnorm,
              ):
    if batchnorm:
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats},
                             batch_inputs, batch_integration_timesteps,
                             )
    else:
        logits = model.apply({"params": state.params},
                             batch_inputs, batch_integration_timesteps,
                             )

    losses = cross_entropy_loss(logits, batch_labels)
    accs = compute_accuracy(logits, batch_labels)

    return losses, accs, logits
