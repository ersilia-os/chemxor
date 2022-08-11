"""Utilities."""

from pathlib import Path
from typing import Any, List, Tuple, Union

import tenseal as ts
import torch as t

from chemxor.schema.fhe_model import PreProcessInput


def get_project_root_path() -> Path:
    """Get path of the project root.

    Returns:
        Path: Project root path
    """
    return Path(__file__).parents[2].absolute()


def prepare_fhe_input(
    model_output: List,
    pre_processors: List[Tuple[PreProcessInput, List]],
    context: ts.Context,
) -> Union[ts.CKKSTensor, ts.CKKSVector]:
    """Prepare input for next step.

    Args:
        model_output (List): Decrypted model ouput to prepare as an input for next step.
        pre_processors (List[Tuple[PreProcessInput, List]]): List of Input pre processor to use
        context (ts.Context): Tenseal encryption context

    Raises:
        NotImplementedError: Pre processor is not implemented

    Returns:
        Union[ts.CKKSTensor, ts.CKKSVector]: Prepared input
    """
    model_output
    for pre_processor in pre_processors:
        if pre_processor[0] == PreProcessInput.IM_TO_COL:
            image_list = []
            for channel in model_output:
                out, _ = ts.im2col_encoding(context, channel, *pre_processor[1])
                image_list.append(out)
            model_output = image_list
        elif pre_processor[0] == PreProcessInput.RE_ENCRYPT:
            model_output = ts.ckks_vector(context, model_output)
        elif pre_processor[0] == PreProcessInput.RESHAPE:
            model_output = t.tensor(model_output).view(pre_processor[1][0]).tolist()
        elif pre_processor[0] == PreProcessInput.PASSTHROUGH:
            model_output = model_output
        else:
            raise NotImplementedError
    return model_output


def evaluate_fhe_model(model: Any, enc_sample: Any, decrypt: bool = False) -> List:
    """Helper to evaluate model on FHE inputs.

    Args:
        model (Any): model wrapped with FHE wrapper
        enc_sample (Any): FHE encrypted sample
        decrypt (bool): decrypt output. Defaults to False.

    Returns:
        List: Model output
    """
    output = enc_sample
    for step in range(model.steps):
        output = model(output, step)
        dec_out = output.decrypt().tolist()
        output = prepare_fhe_input(dec_out, model.pre_process[step], model.enc_context)

    if decrypt is True:
        output = output.decrypt().tolist()
    return output
