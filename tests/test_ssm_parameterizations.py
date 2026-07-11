from types import SimpleNamespace

import jax
import jax.numpy as np

from s5.ssm import init_S5SSM
from s5.ssm_parameterizations import (
    HARDWARE_FRIENDLY_PARAMS,
    init_RealValuedSSM,
    summarize_state_space,
)
from s5.train_helpers import perturb_physical_params


def _args(ssm_param):
    return SimpleNamespace(
        ssm_param=ssm_param,
        ssm_size_base=8,
        d_model=4,
        n_layers=1,
        clip_eigs=False,
        discretization="zoh",
    )


def _find_ssm_module(params):
    if hasattr(params, "items") and "B" in params and "C" in params:
        return params
    for value in params.values():
        if hasattr(value, "items"):
            found = _find_ssm_module(value)
            if found is not None:
                return found
    return None


def _finite_tree(tree):
    return all(bool(np.all(np.isfinite(x))) for x in jax.tree_util.tree_leaves(tree))


def _run_hardware_smoke(ssm_param):
    batch_size, seq_len, input_dim, state_dim = 2, 16, 4, 8
    ssm_cls = init_RealValuedSSM(
        H=input_dim,
        P=state_dim,
        ssm_param=ssm_param,
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
    )
    model = ssm_cls()
    x = np.ones((seq_len, input_dim))
    variables = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(variables, x)
    assert y.shape == (seq_len, input_dim)

    module_params = _find_ssm_module(variables["params"])
    assert module_params is not None
    assert "D" not in module_params
    assert not np.iscomplexobj(module_params["B"])
    assert not np.iscomplexobj(module_params["C"])

    summary = summarize_state_space(variables["params"], _args(ssm_param), use_D=False)
    assert summary["spectral_radius_discrete_A"] < 1.0
    assert summary["B_real"]
    assert summary["C_real"]
    assert not summary["D_enabled"]

    def loss_fn(params):
        out = model.apply({"params": params}, x)
        return np.sum(out ** 2)

    grads = jax.grad(loss_fn)(variables["params"])
    assert _finite_tree(grads)

    batched = jax.vmap(lambda sample: model.apply(variables, sample))
    assert batched(np.ones((batch_size, seq_len, input_dim))).shape == (
        batch_size,
        seq_len,
        input_dim,
    )


def test_hardware_parameterizations_are_stable_real_no_d_and_differentiable():
    for ssm_param in HARDWARE_FRIENDLY_PARAMS:
        _run_hardware_smoke(ssm_param)


def test_physical_noise_perturbed_hardware_params_are_differentiable():
    seq_len, input_dim, state_dim = 16, 4, 8
    x = np.ones((seq_len, input_dim))
    rng = jax.random.PRNGKey(11)

    for ssm_param in HARDWARE_FRIENDLY_PARAMS:
        ssm_cls = init_RealValuedSSM(
            H=input_dim,
            P=state_dim,
            ssm_param=ssm_param,
            discretization="zoh",
            dt_min=0.001,
            dt_max=0.1,
        )
        model = ssm_cls()
        variables = model.init(rng, x)

        def loss_fn(params):
            noisy_params = perturb_physical_params(
                params,
                jax.random.PRNGKey(12),
                0.05,
                ssm_param,
            )
            out = model.apply({"params": noisy_params}, x)
            return np.sum(out ** 2)

        grads = jax.grad(loss_fn)(variables["params"])
        assert _finite_tree(grads)


def test_original_no_d_omits_feedthrough_and_matches_output_shape():
    seq_len, input_dim, state_dim = 16, 4, 8
    eye = np.eye(state_dim, dtype=np.complex64)
    ssm_cls = init_S5SSM(
        H=input_dim,
        P=state_dim,
        Lambda_re_init=-np.ones((state_dim,)),
        Lambda_im_init=np.linspace(0.0, 1.0, state_dim),
        V=eye,
        Vinv=eye,
        C_init="complex_normal",
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=False,
        clip_eigs=False,
        bidirectional=False,
        use_D=False,
    )
    model = ssm_cls()
    x = np.ones((seq_len, input_dim))
    variables = model.init(jax.random.PRNGKey(1), x)
    y = model.apply(variables, x)
    assert y.shape == (seq_len, input_dim)
    assert "D" not in variables["params"]


def test_2x2_parameterizations_require_even_state_dimension():
    ssm_cls = init_RealValuedSSM(
        H=4,
        P=7,
        ssm_param="resonant_2x2",
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
    )
    try:
        ssm_cls().init(jax.random.PRNGKey(2), np.ones((16, 4)))
    except ValueError as exc:
        assert "require an even state dimension" in str(exc)
    else:
        raise AssertionError("Odd state dimension should fail for resonant_2x2.")
