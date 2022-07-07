# ChemXor

Privacy Preserving AI/ML for Drug Discovery

---

## Overview

ChemXor is an open source library for training and evaluating PyTorch models on FHE(Fully homormorphic encryption) encrypted inputs with no manual code changes to the original model. It also provides convenience functions to quickly query and serve these models as a service with strong privacy guarantees for the end user. It is built on top of TenSEAL, Pytorch and ONNX.

### What is Fully Homomorphic Encryption (FHE)?

> A cryptosystem that supports arbitrary computation on ciphertexts is known as fully homomorphic encryption (FHE). Such a scheme enables the construction of programs for any desirable functionality, which can be run on encrypted inputs to produce an encryption of the result. Since such a program need never decrypt its inputs, it can be run by an untrusted party without revealing its inputs and internal state. Fully homomorphic cryptosystems have great practical implications in the outsourcing of private computations. (Wikipedia)

### Why do you need FHE for Machine Learning?

Using FHE, one can compute on encrypted data, without learning anything about the data. This enables novel privacy preserving interactions between actors in the context of machine learning.

## Getting Started

Chemxor is available on PyPi and can be installed using pip.

```bash
pip install chemxor
```

**Encryption context**

We first need to create an encryption context to begin encrypting models and inputs. The TenSeal library is used to create encryption contexts. In the example below, we are using the CKKS encryption scheme.&#x20;

```python
import tenseal as ts

# Create a tenseal context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.global_scale = pow(2, 40)
context.generate_galois_keys()
```

There are other encryption schemes available with their own strengths and weaknesses. Choosing parameters for encryption schemes is not trivial and requires trial and error. More detailed documentation for available encryption schemes and their parameters is available [here](https://github.com/Microsoft/SEAL#examples) and [here](https://github.com/OpenMined/TenSEAL/tree/main/tutorials). There are other projects ([EVA](https://github.com/Microsoft/EVA) compiler for CKKS) that are trying to automate the selection of parameters for specific encryption schemes. However, this is currently out of scope for ChemXor.

**Encrypted Datasets**

ChemXor provides functions to easily convert your Pytorch datasets to Encrypted datasets.

```python
from torch.utils.data import DataLoader
from chemxor.data_modules.enc_dataset import EncDataset

# Use the context that we created earlier
enc_pytorch_dataset = EncDataset(context, pytorch_dataset)

# The encrypted datasets can also be used to create dataloaders
DataLoader(enc_pytorch_dataset, batch_size=None)
```

`EncDataset` class is a wrapper that modifies that **`__getitem__`** method of the `Dataset` class from Pytorch. It encrypts the items using the provided `context` before returning the items.

ChemXor also provides `EncConvDataset` class, a variant to `EncDataset` class for inputs that undergo convolution operations.

```python
from torch.utils.data import DataLoader
from chemxor.data_modules.enc_conv_dataset import EncConvDataset

# Use the context that we created earlier
enc_pytorch_dataset = EncConvDataset(context, pytorch_dataset, kernel_size, stride)

# The encrypted datasets can also be used to create dataloaders
DataLoader(enc_osm_train, batch_size=None)
```

It uses image-to-column encoding of inputs to speed up computation. More details on this topic can be found [here](https://github.com/OpenMined/TenSEAL/blob/main/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb).

> `EncDataset` and `EncConvDataset` does not encrypt the data on disk. Items are encryted lazily on the fly as needed.

**Encrypted models**

ChemXor can automatically convert Pytorch models to models that can be evaluated on encrypted inputs. However, evaluating any arbitrary converted model on encrypted inputs can take an infeasibly long time. This is a major limitation of FHE at the moment.

```python
import tenseal as ts
from chemxor.crypt import FHECryptor
from chemxor.model.cryptic_sage import CrypticSage

# Create a tenseal context
context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=4096,
    plain_modulus=1032193
)

# Initialize the FHECryptor with tenseal context
fhe_cryptor = FHECryptor(context)

# Use any Pytorch Model
model = CrypticSage()

# Convert model using the FHECryptor
converted_model = fhe_cryptor.convert_model(model, dummy_input)

# Converted model is still a Pytorch lightning module
# So use it as usual for evaluating encrypted inputs
enc_output = converted_model(enc_input)
```

ChemXor first converts the Pytorch model to an ONNX model. This ONNX model is then used to create an equivalent function chain that can process encrypted inputs. The resulting converted model is still a Pytorch model with a modified forward function. We are still working on supporting all the operations in the ONNX spec. But, some of the operations might not be available at the time of release.

It is also possible to manually wrap an existing Pytorch model class to make it compatible with encrypted inputs. This is the recommended approach for now as the automatic conversion is not mature yet. There are several models with their encrypted wrappers in ChemXor that can be used as examples.

```python
# Pytorch lightning model
# Adapted from https://github.dev/OpenMined/TenSEAL/blob/6516f215a0171fd9ad70f60f2f9b3d0c83d0d7c4/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb
class ConvNet(pl.LightningModule):
    """Cryptic Sage."""

    def __init__(self: "ConvNet", hidden: int = 64, output: int = 10) -> None:
        """Init."""
        super().__init__()
        self.hidden = hidden
        self.output = output
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, padding=0, stride=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self: "ConvNet", x: Any) -> Any:
        """Forward function.

        Args:
            x (Any): model input

        Returns:
            Any: model output
        """
        x = self.conv1(x)
        # the model uses the square activation function
        x = x * x
        # flattening while keeping the batch axis
        x = x.view(-1, 256)
        x = self.fc1(x)
        x = x * x
        x = self.fc2(x)
        return x

# Encrypted wrapper
# Adapted from https://github.dev/OpenMined/TenSEAL/blob/6516f215a0171fd9ad70f60f2f9b3d0c83d0d7c4/tutorials/Tutorial%204%20-%20Encrypted%20Convolution%20on%20MNIST.ipynb
class EncryptedConvNet(pl.LightningModule):
    """Encrypted ConvNet."""

    def __init__(self: "EncryptedConvNet", model: ConvNet) -> None:
        """Init."""
        super().__init__()

        self.conv1_weight = model.conv1.weight.data.view(
            model.conv1.out_channels,
            model.conv1.kernel_size[0],
            model.conv1.kernel_size[1],
        ).tolist()
        self.conv1_bias = model.conv1.bias.data.tolist()

        self.fc1_weight = model.fc1.weight.T.data.tolist()
        self.fc1_bias = model.fc1.bias.data.tolist()

        self.fc2_weight = model.fc2.weight.T.data.tolist()
        self.fc2_bias = model.fc2.bias.data.tolist()

    def forward(self: "EncryptedConvNet", x: Any, windows_nb: int) -> Any:
        """Forward function.

        Args:
            x (Any): model input
            windows_nb (int): window size.

        Returns:
            Any: model output
        """
        # conv layer
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        # pack all channels into a single flattened vector
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        # square activation
        enc_x.square_()
        # fc1 layer
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        # square activation
        enc_x.square_()
        # fc2 layer
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x
```

A few things to note here:

* We converted Pytorch tensors to a list in the encrypted wrapper. This is required as Pytorch tensors are not compatible with TenSeal encrypted tensors.
* We are not using the standard ReLU activation. CKKS encryption scheme cannot evaluate non-linear piecewise functions. So, either alternative activation functions can be used or polynomial approximations of non-linear activation functions can be used.

#### Serve models

```python
from chemxor.service import create_model_server

# `create_model_server` returns a flask app
flask_app = create_model_server(model, dummy_input)

if __name__ == "__main__":
    flask_app.run()
```

#### Query models

```bash
chemxor query -i [input file path] [model url]
```

#### Distilled models

To overcome the performance limitations of FHE, we are using ChemXor to create simpler distilled models from larger complex models. The distilled models accept inputs as molecules encoded as 32 x 32 images and predict the properties of these molecules. This work is still under progress.

## Developing

We use poetry to manage project dependecies. Use poetry to install project in editable mode.

```bash
poetry install
```

## License

This project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
