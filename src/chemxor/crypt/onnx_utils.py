"""ONNX utilities."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Union


import onnx
import pytorch_lightning as pl
import torch
from torch import nn, Tensor

from chemxor.crypt.fhe_torch_ops import (
    fhe_torch_abs,
    fhe_torch_acos,
    fhe_torch_acosh,
    fhe_torch_add,
    fhe_torch_asin,
    fhe_torch_asinh,
    fhe_torch_atan,
    fhe_torch_atanh,
    fhe_torch_celu,
    fhe_torch_clip,
    fhe_torch_constant,
    fhe_torch_cos,
    fhe_torch_cosh,
    fhe_torch_div,
    fhe_torch_elu,
    fhe_torch_equal,
    fhe_torch_erf,
    fhe_torch_exp,
    fhe_torch_gemm,
    fhe_torch_greater,
    fhe_torch_hardsigmoid,
    fhe_torch_identity,
    fhe_torch_leakyrelu,
    fhe_torch_less,
    fhe_torch_log,
    fhe_torch_matmul,
    fhe_torch_mul,
    fhe_torch_not,
    fhe_torch_relu,
    fhe_torch_reshape,
    fhe_torch_selu,
    fhe_torch_sigmoid,
    fhe_torch_sin,
    fhe_torch_sinh,
    fhe_torch_softplus,
    fhe_torch_sub,
    fhe_torch_tan,
    fhe_torch_tanh,
    fhe_torch_thresholdedrelu,
    torch_conv,
)

# Adapted from https://github.dev/zama-ai/concrete-ml
onnx_to_fhe_torch_dict: Dict[str, Callable] = {
    "Add": fhe_torch_add,
    "Clip": fhe_torch_clip,
    "Constant": fhe_torch_constant,
    "Cos": fhe_torch_cos,
    "Cosh": fhe_torch_cosh,
    "Acos": fhe_torch_acos,
    "Acosh": fhe_torch_acosh,
    "MatMul": fhe_torch_matmul,
    "Gemm": fhe_torch_gemm,
    "Relu": fhe_torch_relu,
    "Selu": fhe_torch_selu,
    "Elu": fhe_torch_elu,
    "Erf": fhe_torch_erf,
    "ThresholdedRelu": fhe_torch_thresholdedrelu,
    "LeakyRelu": fhe_torch_leakyrelu,
    "Celu": fhe_torch_celu,
    "Sin": fhe_torch_sin,
    "Sinh": fhe_torch_sinh,
    "Asin": fhe_torch_asin,
    "Asinh": fhe_torch_asinh,
    "Sigmoid": fhe_torch_sigmoid,
    "HardSigmoid": fhe_torch_hardsigmoid,
    "Tan": fhe_torch_tan,
    "Tanh": fhe_torch_tanh,
    "Atan": fhe_torch_atan,
    "Atanh": fhe_torch_atanh,
    "Softplus": fhe_torch_softplus,
    "Abs": fhe_torch_abs,
    "Div": fhe_torch_div,
    "Mul": fhe_torch_mul,
    "Sub": fhe_torch_sub,
    "Log": fhe_torch_log,
    "Exp": fhe_torch_exp,
    "Equal": fhe_torch_equal,
    "Not": fhe_torch_not,
    "Greater": fhe_torch_greater,
    "Identity": fhe_torch_identity,
    "Reshape": fhe_torch_reshape,
    "Less": fhe_torch_less,
    "Conv": torch_conv,
}

ATTR_TYPES = dict(onnx.AttributeProto.AttributeType.items())
ATTR_GETTERS = {
    ATTR_TYPES["FLOAT"]: lambda attr: attr.f,
    ATTR_TYPES["INT"]: lambda attr: attr.i,
    ATTR_TYPES["STRING"]: lambda attr: attr.s,
    ATTR_TYPES["TENSOR"]: lambda attr: torch.tensor(attr.t),
    ATTR_TYPES["FLOATS"]: lambda attr: attr.floats,
    ATTR_TYPES["INTS"]: lambda attr: tuple(attr.ints),
    ATTR_TYPES["STRINGS"]: lambda attr: attr.strings,
    ATTR_TYPES["TENSORS"]: lambda attr: tuple(
        torch.tensor(val) for val in attr.tensors
    ),
}


# Adapted from https://github.dev/zama-ai/concrete-ml
def execute_onnx_with_fhe_torch(
    onnx_model_graph: onnx.GraphProto, input_tensor: Tensor
) -> Tensor:
    """Execute onnx model using fhe compatible torch functions.

    Args:
        onnx_model_graph (onnx.GraphProto): Onnx model graph
        input_tensor (Tensor): Model input

    Returns:
        Tensor: Model output
    """
    node_results: Dict[str, Tensor] = dict(
        {
            graph_input.name: input_value
            for graph_input, input_value in zip(onnx_model_graph.input, input_tensor)
        },
        **{
            initializer.name: torch.tensor(initializer)
            for initializer in onnx_model_graph.initializer
        },
    )
    for node in onnx_model_graph.node:
        curr_inputs = (node_results[input_name] for input_name in node.input)
        attributes = {
            attribute.name: ATTR_GETTERS[attribute.type](attribute)
            for attribute in node.attribute
        }
        outputs = onnx_to_fhe_torch_dict[node.op_type](*curr_inputs, **attributes)
        node_results.update(zip(node.output, outputs))
    return tuple(node_results[output.name] for output in onnx_model_graph.output)


def torch_to_onnx(
    model: Union[nn.Module, pl.LightningModule], dummy_input: Tensor
) -> Any:
    """Convert Pytorch model to ONNX model.

    Args:
        model (Union[nn.Module, pl.LightningModule]): Pytorch model
        dummy_input (Tensor): Dummy input

    Returns:
        Any: Equivalent ONNX model
    """
    with NamedTemporaryFile() as model_file:
        torch.onnx.export(model, dummy_input, model_file, opset_version=14)
        onnx_model = onnx.load_model(Path(model_file.name))
        onnx.checker.check_model(onnx_model)
    return onnx_model


# Adapted from https://github.dev/zama-ai/concrete-ml/
def onnx_to_torch_fhe_forward(
    onnx_model: onnx.ModelProto, dummpy_input: Tensor
) -> Callable:
    """Use ONNX model to create a FHE compatible torch forward method.

    Args:
        onnx_model (onnx.ModelProto): ONNX model
        dummpy_input (Tensor): Dummy input

    Returns:
        Callable: FHE compatible torch forward method
    """
    return lambda self, input_tensor: execute_onnx_with_fhe_torch(
        onnx_model.graph, input_tensor
    )
