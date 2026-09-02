import jax.numpy as jnp
import numpy as np
import optax
import pytest

from mace_jax.tools.amsgrad import (
    add_decayed_weights_where_grad,
    scale_by_amsgrad,
    scale_by_torch_amsgrad,
    torch_amsgrad,
)


def test_scale_by_amsgrad_update_uses_public_tree_cast():
    params = {'weight': jnp.array([1.0, -2.0], dtype=jnp.float32)}
    gradients = {'weight': jnp.array([0.25, -0.5], dtype=jnp.float32)}
    transform = scale_by_amsgrad(mu_dtype=jnp.float32)

    state = transform.init(params)
    updates, state = transform.update(gradients, state, params)

    np.testing.assert_allclose(updates['weight'], np.array([1.0, -1.0]), rtol=1e-6)
    assert state.count == 1
    assert state.mu['weight'].dtype == jnp.float32


def _torch_reference(params, gradients, learning_rate, b1, b2, eps):
    """NumPy transcription of torch.optim.Adam(..., amsgrad=True)."""
    params = np.asarray(params, dtype=np.float64).copy()
    first = np.zeros_like(params)
    second = np.zeros_like(params)
    second_max = np.zeros_like(params)
    trajectory = []
    for step, gradient in enumerate(gradients, start=1):
        gradient = np.asarray(gradient, dtype=np.float64)
        first = b1 * first + (1.0 - b1) * gradient
        second = b2 * second + (1.0 - b2) * gradient**2
        second_max = np.maximum(second_max, second)
        first_hat = first / (1.0 - b1**step)
        denominator = np.sqrt(second_max / (1.0 - b2**step)) + eps
        params -= learning_rate * first_hat / denominator
        trajectory.append(params.copy())
    return np.asarray(trajectory)


def test_torch_amsgrad_matches_multistep_raw_second_moment_reference():
    b1, b2, eps, learning_rate = 0.9, 0.999, 1.0e-8, 3.0e-3
    gradients = np.asarray([
        [1.0, -0.25],
        [0.4, -0.10],
        [0.1, 0.05],
        [0.02, 0.01],
        [-0.03, 0.005],
    ])
    expected = _torch_reference(
        [0.5, -1.5], gradients, learning_rate, b1, b2, eps
    )
    params = jnp.asarray([0.5, -1.5], dtype=jnp.float32)
    transform = torch_amsgrad(
        learning_rate, b1=b1, b2=b2, eps=eps, mu_dtype=jnp.float32
    )
    state = transform.init(params)
    actual = []
    for gradient in gradients:
        updates, state = transform.update(jnp.asarray(gradient), state, params)
        params = params + updates
        actual.append(np.asarray(params))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)


def test_torch_amsgrad_maximum_is_raw_not_bias_corrected():
    transform = scale_by_torch_amsgrad(b1=0.9, b2=0.999)
    params = {'weight': jnp.asarray([1.0], dtype=jnp.float32)}
    state = transform.init(params)
    for gradient in (1.0, 0.1, 0.01):
        _, state = transform.update(
            {'weight': jnp.asarray([gradient], dtype=jnp.float32)},
            state,
            params,
        )
    raw_second = float(state.nu['weight'][0])
    raw_max = float(state.nu_max['weight'][0])
    assert raw_max >= raw_second
    # A bias-corrected maximum would be O(1) after the first unit gradient;
    # PyTorch's raw maximum remains O(1-b2).
    assert raw_max < 0.01


def test_torch_amsgrad_matches_pytorch_with_coupled_weight_decay():
    """Exercise the long-run decay direction that exposed the OOD bug."""
    torch = pytest.importorskip("torch")
    learning_rate = 1.0e-3
    weight_decay = 5.0e-4
    initial = np.asarray([1.0, -0.3, 0.02], dtype=np.float32)

    torch_params = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    torch_optimizer = torch.optim.Adam(
        [torch_params],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=weight_decay,
        amsgrad=True,
    )

    jax_params = jnp.asarray(initial)
    transform = optax.chain(
        optax.add_decayed_weights(weight_decay),
        torch_amsgrad(learning_rate, b1=0.9, b2=0.999, eps=1.0e-8),
    )
    state = transform.init(jax_params)

    # The data-gradient magnitude decays and changes sign, so the raw-vs-
    # bias-corrected maximum distinction is exercised in addition to coupled
    # L2 decay.  One-step tests cannot detect that distinction.
    for step in range(1, 1001):
        data_gradient = np.asarray(
            [
                0.03 * np.exp(-step / 80.0),
                -0.01 * np.exp(-step / 200.0),
                0.004 * np.cos(step / 19.0) * np.exp(-step / 300.0),
            ],
            dtype=np.float32,
        )
        torch_optimizer.zero_grad(set_to_none=True)
        torch_params.grad = torch.from_numpy(data_gradient.copy())
        torch_optimizer.step()

        updates, state = transform.update(
            jnp.asarray(data_gradient), state, jax_params
        )
        jax_params = optax.apply_updates(jax_params, updates)

    np.testing.assert_allclose(
        np.asarray(jax_params),
        torch_params.detach().numpy(),
        rtol=2.0e-5,
        atol=2.0e-7,
    )


def test_gated_decay_freezes_identically_zero_grad_leaf():
    """A leaf with exact-zero grads must stay bitwise frozen (torch None-grad
    semantics); a live leaf must match the ungated decay chain exactly."""
    learning_rate = 1.0e-3
    weight_decay = 5.0e-4
    live0 = jnp.asarray([1.0, -0.3], dtype=jnp.float32)
    dead0 = jnp.asarray([0.7, 0.2], dtype=jnp.float32)

    params = {'live': live0, 'dead': dead0}
    gated = optax.chain(
        add_decayed_weights_where_grad(weight_decay),
        torch_amsgrad(learning_rate),
    )
    state = gated.init(params)

    solo = jnp.asarray(live0)
    ungated = optax.chain(
        optax.add_decayed_weights(weight_decay),
        torch_amsgrad(learning_rate),
    )
    solo_state = ungated.init(solo)

    for step in range(1, 6):
        live_grad = jnp.asarray(
            [0.02 / step, -0.01 * step], dtype=jnp.float32)
        grads = {'live': live_grad,
                 'dead': jnp.zeros_like(dead0)}
        updates, state = gated.update(grads, state, params)
        params = optax.apply_updates(params, updates)

        solo_updates, solo_state = ungated.update(live_grad, solo_state, solo)
        solo = optax.apply_updates(solo, solo_updates)

    assert np.array_equal(np.asarray(params['dead']), np.asarray(dead0)), (
        'zero-grad leaf must be bitwise unchanged')
    np.testing.assert_array_equal(np.asarray(params['live']), np.asarray(solo))


def test_gated_decay_matches_pytorch_none_grad_param():
    """1000 steps vs real torch.optim.Adam with one param whose grad stays
    None (set_to_none=True and never assigned): torch applies neither decay
    nor moments to it; the gated chain must reproduce both params."""
    torch = pytest.importorskip("torch")
    learning_rate = 1.0e-3
    weight_decay = 5.0e-4
    used0 = np.asarray([1.0, -0.3, 0.02], dtype=np.float32)
    unused0 = np.asarray([0.5, -0.8], dtype=np.float32)

    p_used = torch.nn.Parameter(torch.from_numpy(used0.copy()))
    p_unused = torch.nn.Parameter(torch.from_numpy(unused0.copy()))
    torch_optimizer = torch.optim.Adam(
        [p_used, p_unused],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=weight_decay,
        amsgrad=True,
    )

    jax_params = {'used': jnp.asarray(used0), 'unused': jnp.asarray(unused0)}
    transform = optax.chain(
        add_decayed_weights_where_grad(weight_decay),
        torch_amsgrad(learning_rate, b1=0.9, b2=0.999, eps=1.0e-8),
    )
    state = transform.init(jax_params)

    for step in range(1, 1001):
        data_gradient = np.asarray(
            [
                0.03 * np.exp(-step / 80.0),
                -0.01 * np.exp(-step / 200.0),
                0.004 * np.cos(step / 19.0) * np.exp(-step / 300.0),
            ],
            dtype=np.float32,
        )
        torch_optimizer.zero_grad(set_to_none=True)
        p_used.grad = torch.from_numpy(data_gradient.copy())
        torch_optimizer.step()

        grads = {'used': jnp.asarray(data_gradient),
                 'unused': jnp.zeros_like(jax_params['unused'])}
        updates, state = transform.update(grads, state, jax_params)
        jax_params = optax.apply_updates(jax_params, updates)

    np.testing.assert_array_equal(
        np.asarray(jax_params['unused']), unused0)
    np.testing.assert_array_equal(
        p_unused.detach().numpy(), unused0)
    np.testing.assert_allclose(
        np.asarray(jax_params['used']),
        p_used.detach().numpy(),
        rtol=2.0e-5,
        atol=2.0e-7,
    )
