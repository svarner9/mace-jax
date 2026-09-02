# ###Functions taken from the PR of @merajhashemi d5d72605a5 ####

from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics, utils

_abs_sq = numerics.abs_sq


def update_moment(updates, moments, decay, order):
    """Compute the exponential moving average of the `order`-th moment."""
    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * (g**order) + decay * t, updates, moments
    )


def update_moment_per_elem_norm(updates, moments, decay, order):
    """Compute the EMA of the `order`-th moment of the element-wise norm."""

    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g**order
        else:
            half_order = order / 2
            # JAX generates different HLO for int and float `order`
            if half_order.is_integer():
                half_order = int(half_order)
            return _abs_sq(g) ** half_order

    return jax.tree_util.tree_map(
        lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments
    )


def bias_correction(moment, decay, count):
    """Perform bias correction. This becomes a no-op as count goes to infinity."""
    bias_correction_ = 1 - decay**count
    return jax.tree_util.tree_map(
        lambda t: t / bias_correction_.astype(t.dtype), moment
    )


class ScaleByAmsgradState(NamedTuple):
    """State for the AMSGrad algorithm."""

    count: chex.Array  # shape=(), dtype=jnp.int32.
    mu: base.Updates
    nu: base.Updates
    nu_max: base.Updates


class ScaleByTorchAmsgradState(NamedTuple):
    """State for PyTorch-compatible AMSGrad.

    PyTorch stores the elementwise maximum of the *raw* second moment and
    applies the current step's bias correction only when forming the
    denominator.  Optax instead stores the maximum of the bias-corrected
    second moment.  The distinction is large for decaying gradients such as
    coupled L2 decay, despite being invisible on the first update.
    """

    count: chex.Array
    mu: base.Updates
    nu: base.Updates
    nu_max: base.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    """Rescale updates according to the AMSGrad algorithm.
    References:
        [Reddi et al, 2018](https://openreview.net/forum?id=ryQu7f-RZ)
    Args:
        b1: decay rate for the exponentially weighted average of grads.
        b2: decay rate for the exponentially weighted average of squared grads.
        eps: term added to the denominator to improve numerical stability.
        eps_root: term added to the denominator inside the square-root to improve
          numerical stability when backpropagating gradients through the rescaling.
        mu_dtype: optional `dtype` to be used for the first order accumulator; if
          `None` then the `dtype is inferred from `params` and `updates`.
    Returns:
        An (init_fn, update_fn) tuple.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(  # First moment
            lambda t: jnp.zeros_like(t, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        nu_max = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByAmsgradState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_max=nu_max
        )

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_hat = bias_correction(nu, b2, count_inc)
        nu_max = jax.tree_util.tree_map(jnp.maximum, state.nu_max, nu_hat)
        updates = jax.tree_util.tree_map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_max
        )
        # ``optax._src.utils.cast_tree`` was removed in Optax 0.2.8.  Use the
        # public tree utility so AMSGrad works across current Optax releases.
        mu = optax.tree_utils.tree_cast(mu, mu_dtype)
        return updates, ScaleByAmsgradState(
            count=count_inc, mu=mu, nu=nu, nu_max=nu_max
        )

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_torch_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    """Rescale updates with the exact ``torch.optim.Adam(amsgrad=True)`` rule.

    In particular, ``nu_max = max(nu_max, nu)`` is evaluated before bias
    correction.  At step ``t`` the denominator is then

    ``sqrt(nu_max / (1 - b2**t) + eps_root) + eps``.

    This ordering is the production MACE optimizer contract.  It is not the
    ordering used by :func:`optax.amsgrad` or :func:`scale_by_amsgrad` above.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(value, dtype=mu_dtype), params
        )
        nu = jax.tree_util.tree_map(jnp.zeros_like, params)
        nu_max = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByTorchAmsgradState(
            count=jnp.zeros([], jnp.int32), mu=mu, nu=nu, nu_max=nu_max
        )

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        nu_max = jax.tree_util.tree_map(jnp.maximum, state.nu_max, nu)
        nu_max_hat = bias_correction(nu_max, b2, count_inc)
        scaled = jax.tree_util.tree_map(
            lambda first, second: first
            / (jnp.sqrt(second + eps_root) + eps),
            mu_hat,
            nu_max_hat,
        )
        mu = optax.tree_utils.tree_cast(mu, mu_dtype)
        return scaled, ScaleByTorchAmsgradState(
            count=count_inc, mu=mu, nu=nu, nu_max=nu_max
        )

    return base.GradientTransformation(init_fn, update_fn)


def torch_amsgrad(
    learning_rate,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Any | None = None,
) -> optax.GradientTransformation:
    """Full PyTorch-compatible AMSGrad optimizer for Optax pipelines."""

    return optax.chain(
        scale_by_torch_amsgrad(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
        ),
        optax.scale_by_learning_rate(learning_rate),
    )


def zero_nonfinite() -> optax.GradientTransformation:
    """Replace every non-finite gradient entry (NaN **or Inf**) with zero.

    Place this FIRST in the chain, ahead of any global-norm clip.
    ``optax.clip_by_global_norm`` computes ``g * min(1, max_norm/||g||)``; a
    single bad leaf makes ``||g||`` non-finite, so the scale is non-finite and
    the clip multiplies it into EVERY parameter. Clipping bounds large *finite*
    gradients — it launders non-finite ones into the weights. Zeroing them here
    turns a poisoned step into a no-op and leaves the last good weights intact.

    ``optax.zero_nans()`` is NOT sufficient: it handles NaN only, and an Inf
    gradient still kills the run because ``max_norm/Inf -> 0`` and then
    ``Inf * 0 -> NaN``. Verified: with zero_nans the Inf case still produced
    non-finite parameters; with this transform it does not.

    Identity on finite gradients, so healthy runs are bit-identical (verified:
    ``max|new-old| == 0`` for both ordinary and clip-saturating gradients).
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g)),
            updates)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def add_decayed_weights_where_grad(
    weight_decay: float,
) -> optax.GradientTransformation:
    """Coupled L2 decay gated per leaf on the incoming gradient being nonzero.

    ``torch.optim.Adam`` skips parameters whose ``.grad`` is ``None``
    entirely — no coupled decay, no moment update.  A JAX module that never
    executes produces exact-zero gradients instead of ``None``, so ungated
    ``optax.add_decayed_weights`` would decay such a leaf every step while
    torch leaves it bitwise-frozen (observed: the never-executed m2m/l2l TP
    ground to RMS 7.7e-4 vs torch's untouched 0.99988 on MFN cumulene).
    Gating on ``any(g != 0)`` restores torch semantics: with a zero gradient
    the AMSGrad update is exactly zero, so the leaf never moves.

    Caveat (documented contract difference): a parameter that trained earlier
    and then receives a zero-grad step keeps its torch per-param step count
    frozen, while the optax global count advances; the next nonzero step's
    bias correction differs slightly.  Deep production trees (water) have
    every family active on every batch, so this intermittent case does not
    arise there.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError(
                'add_decayed_weights_where_grad requires params')

        def one(g, p):
            gate = jnp.any(g != 0).astype(g.dtype)
            return g + weight_decay * gate * p

        return jax.tree_util.tree_map(one, updates, params), state

    return base.GradientTransformation(init_fn, update_fn)
