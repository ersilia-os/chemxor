"""FHE model wrappers."""

from typing import Any

import pytorch_lightning as pl
import tenseal as ts

from chemxor.model.fhe_activation import softplus_polyval
from chemxor.model.olinda_net import OlindaNet, OlindaNetOne, OlindaNetZero
from chemxor.schema.fhe_model import PreProcessInput


class FHEOlindaNetZero(pl.LightningModule):
    """FHEOlindaNet Zero: Wrapper to evaluate FHE inputs."""

    def __init__(self: "FHEOlindaNetZero", model: OlindaNetZero) -> None:
        """Init."""
        super().__init__()

        self._model = model

        # Prepare layers
        self.conv1_weight = model.conv1.weight.data.view(
            model.conv1.in_channels,
            model.conv1.out_channels,
            model.conv1.kernel_size[0],
            model.conv1.kernel_size[1],
        ).tolist()
        self.conv1_bias = model.conv1.bias.data.tolist()

        self.fc1_weight = model.fc1.weight.T.data.tolist()
        self.fc1_bias = model.fc1.bias.data.tolist()

        self.fc2_weight = model.fc2.weight.T.data.tolist()
        self.fc2_bias = model.fc2.bias.data.tolist()

        self.fc3_weight = model.fc3.weight.T.data.tolist()
        self.fc3_bias = model.fc3.bias.data.tolist()

        # Prepare parameters
        self.steps = 2
        self.conv1_windows_nb = 100
        self.pre_process = [
            [(PreProcessInput.RESHAPE, [3200]), (PreProcessInput.RE_ENCRYPT, [])],
            [(PreProcessInput.RE_ENCRYPT, [])],
            [(PreProcessInput.PASSTHROUGH, [])],
        ]

        # Encryption context
        bits_scale = 26
        self.enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[
                31,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                31,
            ],
        )
        self.enc_context.global_scale = pow(2, bits_scale)
        self.enc_context.generate_galois_keys()

    def forward(self: "FHEOlindaNetZero", x: Any, step: int) -> Any:
        """Forward function.

        Args:
            x (Any): model input
            step (int): model step

        Returns:
            Any: model output
        """
        if step == 0:
            # conv layer 1
            running_sum = None
            for i, _ in enumerate(self.conv1_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv1_weight[i], self.conv1_bias):
                    y = x.conv2d_im2col(kernel, self.conv1_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 1:
            # fc1 layer
            enc_x = x.mm(self.fc1_weight) + self.fc1_bias
            enc_x = softplus_polyval(enc_x)
            # fc2 layer
            enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 2:
            # fc3 layer
            enc_x = x.mm(self.fc3_weight) + self.fc3_bias
            return enc_x


class FHEOlindaNet(pl.LightningModule):
    """FHEOlindaNet: Wrapper to evaluate FHE inputs."""

    def __init__(self: "FHEOlindaNet", model: OlindaNet) -> None:
        """Init."""
        super().__init__()

        self._model = model

        # Prepare layers
        self.conv1_weight = model.conv1.weight.data.view(
            model.conv1.in_channels,
            model.conv1.out_channels,
            model.conv1.kernel_size[0],
            model.conv1.kernel_size[1],
        ).tolist()
        self.conv1_bias = model.conv1.bias.data.tolist()

        self.conv2_weight = model.conv2.weight.data.view(
            model.conv1.in_channels,
            model.conv2.out_channels,
            model.conv2.kernel_size[0],
            model.conv2.kernel_size[1],
        ).tolist()
        self.conv2_bias = model.conv2.bias.data.tolist()

        self.fc1_weight = model.fc1.weight.T.data.tolist()
        self.fc1_bias = model.fc1.bias.data.tolist()

        self.fc2_weight = model.fc2.weight.T.data.tolist()
        self.fc2_bias = model.fc2.bias.data.tolist()

        self.fc3_weight = model.fc3.weight.T.data.tolist()
        self.fc3_bias = model.fc3.bias.data.tolist()

        self.fc4_weight = model.fc4.weight.T.data.tolist()
        self.fc4_bias = model.fc4.bias.data.tolist()

        # Prepare parameters
        self.steps = 3
        self.conv1_windows_nb = 100
        self.conv2_windows_nb = 64

        self.pre_process = [
            [
                (PreProcessInput.RESHAPE, [(32, 10, 10)]),
                (PreProcessInput.IM_TO_COL, [3, 3, 1]),
            ],
            [(PreProcessInput.RESHAPE, [3200]), (PreProcessInput.RE_ENCRYPT, [])],
            [(PreProcessInput.RE_ENCRYPT, [])],
            [(PreProcessInput.PASSTHROUGH, [])],
        ]

        # Encryption context
        bits_scale = 26
        self.enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[
                31,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                31,
            ],
        )
        self.enc_context.global_scale = pow(2, bits_scale)
        self.enc_context.generate_galois_keys()

    def forward(self: "FHEOlindaNet", x: Any, step: int) -> Any:
        """Forward function.

        Args:
            x (Any): model input
            step (int): model step

        Returns:
            Any: model output
        """
        if step == 0:
            # conv layer 1
            running_sum = None
            for i, _ in enumerate(self.conv1_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv1_weight[i], self.conv1_bias):
                    y = x.conv2d_im2col(kernel, self.conv1_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 1:
            # conv layer 2
            running_sum = None
            for i, _ in enumerate(self.conv2_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv2_weight[i], self.conv2_bias):
                    y = x[i].conv2d_im2col(kernel, self.conv2_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 2:
            # fc1 layer
            enc_x = x.mm(self.fc1_weight) + self.fc1_bias
            enc_x = softplus_polyval(enc_x)
            # fc2 layer
            enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
            enc_x = softplus_polyval(enc_x)
            return enc_x

        elif step == 3:
            # fc3 layer
            enc_x = x.mm(self.fc3_weight) + self.fc3_bias
            enc_x = softplus_polyval(enc_x)
            # fc4 layer
            enc_x = enc_x.mm(self.fc4_weight) + self.fc4_bias
            return enc_x


class FHEOlindaNetOne(pl.LightningModule):
    """FHEOlindaNet One: Wrapper to evaluate FHE inputs."""

    def __init__(self: "FHEOlindaNetOne", model: OlindaNetOne) -> None:
        """Init."""
        super().__init__()

        self._model = model

        # Prepare layers
        self.conv1_weight = model.conv1.weight.data.view(
            model.conv1.in_channels,
            model.conv1.out_channels,
            model.conv1.kernel_size[0],
            model.conv1.kernel_size[1],
        ).tolist()
        self.conv1_bias = model.conv1.bias.data.tolist()

        self.conv2_weight = model.conv2.weight.data.view(
            model.conv2.in_channels,
            model.conv2.out_channels,
            model.conv2.kernel_size[0],
            model.conv2.kernel_size[1],
        ).tolist()
        self.conv2_bias = model.conv2.bias.data.tolist()

        self.conv3_weight = model.conv3.weight.data.view(
            model.conv3.in_channels,
            model.conv3.out_channels,
            model.conv3.kernel_size[0],
            model.conv3.kernel_size[1],
        ).tolist()
        self.conv3_bias = model.conv3.bias.data.tolist()

        self.conv4_weight = model.conv4.weight.data.view(
            model.conv4.in_channels,
            model.conv4.out_channels,
            model.conv4.kernel_size[0],
            model.conv4.kernel_size[1],
        ).tolist()
        self.conv4_bias = model.conv4.bias.data.tolist()

        self.fc1_weight = model.fc1.weight.T.data.tolist()
        self.fc1_bias = model.fc1.bias.data.tolist()

        self.fc2_weight = model.fc2.weight.T.data.tolist()
        self.fc2_bias = model.fc2.bias.data.tolist()

        self.fc3_weight = model.fc3.weight.T.data.tolist()
        self.fc3_bias = model.fc3.bias.data.tolist()

        self.fc4_weight = model.fc4.weight.T.data.tolist()
        self.fc4_bias = model.fc4.bias.data.tolist()

        # Prepare parameters
        self.steps = 5
        self.conv1_windows_nb = 100
        self.conv2_windows_nb = 64
        self.conv3_windows_nb = 36
        self.conv4_windows_nb = 16

        self.pre_process = [
            [
                (PreProcessInput.RESHAPE, [(32, 10, 10)]),
                (PreProcessInput.IM_TO_COL, [3, 3, 1]),
            ],
            [
                (PreProcessInput.RESHAPE, [(32, 8, 8)]),
                (PreProcessInput.IM_TO_COL, [3, 3, 1]),
            ],
            [
                (PreProcessInput.RESHAPE, [(64, 6, 6)]),
                (PreProcessInput.IM_TO_COL, [3, 3, 1]),
            ],
            [(PreProcessInput.RESHAPE, [(1024)]), (PreProcessInput.RE_ENCRYPT, [])],
            [(PreProcessInput.RE_ENCRYPT, [])],
            [(PreProcessInput.PASSTHROUGH, [])],
        ]

        # Encryption context
        bits_scale = 26
        self.enc_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[
                31,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                bits_scale,
                31,
            ],
        )
        self.enc_context.global_scale = pow(2, bits_scale)
        self.enc_context.generate_galois_keys()

    def forward(self: "FHEOlindaNetOne", x: Any, step: int) -> Any:
        """Forward function.

        Args:
            x (Any): model input
            step (int): model step

        Returns:
            Any: model output
        """
        if step == 0:
            # conv layer 1
            running_sum = None
            for i, _ in enumerate(self.conv1_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv1_weight[i], self.conv1_bias):
                    y = x.conv2d_im2col(kernel, self.conv1_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 1:
            # conv layer 2
            running_sum = None
            for i, _ in enumerate(self.conv2_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv2_weight[i], self.conv2_bias):
                    y = x[i].conv2d_im2col(kernel, self.conv2_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 2:
            # conv layer 3
            running_sum = None
            for i, _ in enumerate(self.conv3_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv3_weight[i], self.conv3_bias):
                    y = x[i].conv2d_im2col(kernel, self.conv3_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 3:
            # conv layer 4
            running_sum = None
            for i, _ in enumerate(self.conv4_weight):
                enc_channels = []
                for kernel, bias in zip(self.conv4_weight[i], self.conv4_bias):
                    y = x[i].conv2d_im2col(kernel, self.conv4_windows_nb) + bias
                    enc_channels.append(y)
                # pack all channels into a single flattened vector
                enc_x = ts.CKKSVector.pack_vectors(enc_channels)
                if running_sum is None:
                    running_sum = enc_x
                else:
                    running_sum = running_sum + enc_x
            enc_x = softplus_polyval(enc_x)
            return enc_x
        elif step == 4:
            # fc1 layer
            enc_x = x.mm(self.fc1_weight) + self.fc1_bias
            enc_x = softplus_polyval(enc_x)
            # fc2 layer
            enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
            enc_x = softplus_polyval(enc_x)
            return enc_x

        elif step == 5:
            # fc3 layer
            enc_x = x.mm(self.fc3_weight) + self.fc3_bias
            enc_x = softplus_polyval(enc_x)
            # fc4 layer
            enc_x = enc_x.mm(self.fc4_weight) + self.fc4_bias
            return enc_x
