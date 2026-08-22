"""
Straight-Through Estimator (STE) for differentiable quantization.

The STE allows gradient-based optimisation of quantized parameters by
using the identity function on the backward pass while applying the
non-differentiable quantization step on the forward pass.

Reference
---------
Bengio, Léonard & Courville, "Estimating or Propagating Gradients Through
Stochastic Neurons for Conditional Computation," arXiv:1308.3432, 2013.
"""

import torch
from torch.autograd import Function


class _STEFunction(Function):
    """
    Custom autograd function implementing the Straight-Through Estimator.

    Forward:  y = quantize(x)   (discrete output)
    Backward: ∂L/∂x = ∂L/∂y    (gradient passes through unchanged)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, x_quantized: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returns the pre-computed quantized value.

        Args:
            x: Continuous input (for gradient routing only).
            x_quantized: Quantized version of x (detached from graph).

        Returns:
            x_quantized, with gradient routed back through x.
        """
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: pass gradient directly to the continuous input.

        Args:
            grad_output: Upstream gradient ∂L/∂y.

        Returns:
            (∂L/∂x, None): Gradient for x; no gradient for x_quantized.
        """
        return grad_output, None


class StraightThroughEstimator(torch.nn.Module):
    """
    Module wrapper around the STE autograd function.

    Applies a supplied quantization function on the forward pass while
    routing gradients straight through on the backward pass.

    Args:
        quantize_fn: Callable that maps continuous tensor → quantized tensor.
            Must be a pure function operating element-wise (no learnable
            parameters of its own).

    Example::

        >>> import torch
        >>> from d2nn.quantization.ste import StraightThroughEstimator
        >>> quant_fn = lambda x: torch.round(x)
        >>> ste = StraightThroughEstimator(quant_fn)
        >>> x = torch.tensor([0.3, 1.7, 2.5], requires_grad=True)
        >>> y = ste(x)
        >>> y  # tensor([0., 2., 2.])
        >>> y.sum().backward()
        >>> x.grad  # tensor([1., 1., 1.])  — gradients pass through
    """

    def __init__(self, quantize_fn):
        super().__init__()
        self.quantize_fn = quantize_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.quantize_fn(x)
        return _STEFunction.apply(x, x_q)


def ste_quantize(x: torch.Tensor, x_quantized: torch.Tensor) -> torch.Tensor:
    """
    Functional API: apply STE to a pair of (continuous, quantized) tensors.

    Equivalent to ``_STEFunction.apply(x, x_quantized)`` but with a
    friendlier public name.

    Args:
        x: Continuous-valued tensor (carries the gradient).
        x_quantized: Quantized tensor (returned on forward pass).

    Returns:
        ``x_quantized`` in the forward graph, with gradients routed to ``x``.
    """
    return _STEFunction.apply(x, x_quantized)
