"""FHE torch OPS."""

from typing import Optional, Tuple, Union

import tenseal as ts
import torch


def fhe_torch_add() -> Tuple:
    pass


def fhe_torch_abs():
    pass


def fhe_torch_acos():
    pass


def fhe_torch_acosh():
    pass


def fhe_torch_asin():
    pass


def fhe_torch_asinh():
    pass


def fhe_torch_atan():
    pass


def fhe_torch_atanh():
    pass


def fhe_torch_celu():
    pass


def fhe_torch_clip():
    pass


def fhe_torch_constant():
    pass


def fhe_torch_cos():
    pass


def fhe_torch_cosh():
    pass


def fhe_torch_div():
    pass


def fhe_torch_elu():
    pass


def fhe_torch_equal():
    pass


def fhe_torch_erf():
    pass


def fhe_torch_exp():
    pass


# priority
# transA and transB are not snake case but need to match ONNX attribute naming, ignore the lint
# pylint: disable=invalid-name
# 1 is technically an int but is accepted by mypy as a float (and it simplifies our life for
# compilation) so instead of passing 1.0 by default 1 is passed
def fhe_torch_gemm(
    a: ts.CKKSTensor,
    b: torch.Tensor,
    /,
    c: Optional[torch.Tensor] = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    transA: int = 0,
    transB: int = 0,
) -> Tuple[ts.CKKSTensor]:
    """Compute Gemm in ts.CKKSTensor according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-13

    Args:
        a (ts.CKKSTensor): Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M)
            if transA is non-zero.
        b (torch.Tensor): Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K)
            if transB is non-zero.
        c (Optional[torch.Tensor]): Optional input tensor C. If not specified, the
            computation is done as if C is a scalar 0. The shape of C should be unidirectional
            broadcastable to (M, N).
            Defaults to None.
        alpha (float): Scalar multiplier for the product of input tensors A * B.
            Defaults to 1.
        beta (float): Scalar multiplier for input tensor C.
            Defaults to 1.
        transA (int): Whether A should be transposed. The type is kept as int as it's the
            type used by ONNX and it can easily be interpreted by python as a boolean.
            Defaults to 0.
        transB (int): Whether B should be transposed. The type is kept as int as it's the
            type used by ONNX and it can easily be interpreted by python as a boolean.
            Defaults to 0.

    Returns:
        Tuple[ts.CKKSTensor]: The tuple containing the result tensor
    """
    # If alpha and beta are integer, apply the int type for concrete-numpy
    # to see they are integers (see issue #277)
    processed_alpha = int(alpha) if round(alpha) == alpha else alpha
    processed_beta = int(beta) if round(beta) == beta else beta

    a_prime = a.transpose() if transA else a
    b_prime = b.transpose() if transB else b
    c_prime: Union[torch.Tensor, float] = c if c is not None else 0

    # Do
    #
    #       y = processed_alpha * numpy.matmul(a_prime, b_prime) + processed_beta * c_prime
    #
    # in an efficient way, ie to make tracing directly optimized, without expecting any opt from the
    # compiler here

    y = a_prime.mm(b_prime)

    if processed_alpha != 1:
        y = y * processed_alpha

    # fixme: elementeise comparison is required
    if c_prime != 0:
        if processed_beta == 1:
            y = y + c_prime
        else:
            y = y + processed_beta * c_prime

    return (y,)


def fhe_torch_greater():
    pass


def fhe_torch_hardsigmoid():
    pass


def fhe_torch_identity():
    pass


def fhe_torch_leakyrelu():
    pass


def fhe_torch_less():
    pass


def fhe_torch_log():
    pass


def fhe_torch_matmul():
    pass


def fhe_torch_mul():
    pass


def fhe_torch_not():
    pass


# priority
def fhe_torch_relu(x: ts.CKKSTensor, /) -> Tuple[ts.CKKSTensor]:
    """Relu equivalent in FHE.

    Args:
        x (ts.CKKSTensor): Input tensor

    Returns:
        Tuple[ts.CKKSTensor]: Output tensor
    """
    return x.square_()


# priority
def fhe_torch_reshape(
    x: ts.CKKSTensor, newshape: Union[list, torch.Tensor], /, *, allowzero=0
) -> Tuple[ts.CKKSTensor]:
    """Compute reshape in ts.CKKSTensor according to ONNX spec.

    Args:
        x (ts.CKKSTensor): input tensor
        newshape (Union[list, torch.Tensor]): new shape
        allowzero (int, optional): _description_. Defaults to 0.

    Returns:
        Tuple[ts.CKKSTensor]: reshaped tensor
    """

    return (x.reshape(newshape),)


def fhe_torch_selu():
    pass


def fhe_torch_sigmoid():
    pass


def fhe_torch_sin():
    pass


def fhe_torch_sinh():
    pass


def fhe_torch_softplus():
    pass


def fhe_torch_sub():
    pass


def fhe_torch_tan():
    pass


def fhe_torch_tanh():
    pass


def fhe_torch_thresholdedrelu():
    pass


def torch_conv():
    pass
