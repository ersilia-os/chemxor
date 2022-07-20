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
The direct and indirect dependecies of the project are licensed as follows:

| Name                      | Version     | License                                                                                             | Author                                                                                                           |
|---------------------------|-------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Babel                     | 2.10.1      | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| CacheControl              | 0.12.11     | Apache Software License                                                                             | Eric Larson                                                                                                      |
| Faker                     | 13.12.0     | MIT License                                                                                         | joke2k                                                                                                           |
| Flask                     | 2.1.2       | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| GitPython                 | 3.1.27      | BSD License                                                                                         | Sebastian Thiel, Michael Trier                                                                                   |
| Jinja2                    | 3.1.2       | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| Markdown                  | 3.3.7       | BSD License                                                                                         | Manfred Stienstra, Yuri takhteyev and Waylan limberg                                                             |
| MarkupSafe                | 2.1.1       | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| Pillow                    | 9.1.1       | Historical Permission Notice and Disclaimer (HPND)                                                  | Alex Clark (PIL Fork Author)                                                                                     |
| PyJWT                     | 2.3.0       | MIT License                                                                                         | Jose Padilla                                                                                                     |
| PyNaCl                    | 1.4.0       | Apache License 2.0                                                                                  | The PyNaCl developers                                                                                            |
| PyYAML                    | 5.4.1       | MIT License                                                                                         | Kirill Simonov                                                                                                   |
| Pygments                  | 2.12.0      | BSD License                                                                                         | Georg Brandl                                                                                                     |
| SQLAlchemy                | 1.4.36      | MIT License                                                                                         | Mike Bayer                                                                                                       |
| SecretStorage             | 3.3.2       | BSD License                                                                                         | Dmitry Shachnev                                                                                                  |
| Send2Trash                | 1.8.0       | BSD License                                                                                         | Andrew Senetar                                                                                                   |
| Werkzeug                  | 2.1.2       | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| absl-py                   | 1.0.0       | Apache Software License                                                                             | The Abseil Authors                                                                                               |
| aiofiles                  | 0.6.0       | Apache Software License                                                                             | Tin Tvrtkovic                                                                                                    |
| aiohttp                   | 3.8.1       | Apache Software License                                                                             | UNKNOWN                                                                                                          |
| aiosignal                 | 1.2.0       | Apache Software License                                                                             | Nikolay Kim                                                                                                      |
| anyconfig                 | 0.10.1      | MIT License                                                                                         | Satoru SATOH                                                                                                     |
| anyio                     | 3.6.1       | MIT License                                                                                         | Alex Grönholm                                                                                                    |
| argcomplete               | 1.12.3      | Apache Software License                                                                             | Andrey Kislyuk                                                                                                   |
| argon2-cffi               | 21.3.0      | MIT License                                                                                         | UNKNOWN                                                                                                          |
| argon2-cffi-bindings      | 21.2.0      | MIT License                                                                                         | Hynek Schlawack                                                                                                  |
| arrow                     | 1.2.2       | Apache Software License                                                                             | Chris Smith                                                                                                      |
| ascii-magic               | 1.6         | MIT License                                                                                         | Leandro Barone                                                                                                   |
| ase                       | 3.22.1      | GNU Lesser General Public License v2 or later (LGPLv2+)                                             | UNKNOWN                                                                                                          |
| asgiref                   | 3.5.2       | BSD License                                                                                         | Django Software Foundation                                                                                       |
| async-timeout             | 4.0.2       | Apache Software License                                                                             | Andrew Svetlov <andrew.svetlov@gmail.com>                                                                        |
| attrs                     | 21.4.0      | MIT License                                                                                         | Hynek Schlawack                                                                                                  |
| autodp                    | 0.2         | Apache Software License                                                                             | Yu-Xiang Wang                                                                                                    |
| backcall                  | 0.2.0       | BSD License                                                                                         | Thomas Kluyver                                                                                                   |
| backports.cached-property | 1.0.1       | MIT License                                                                                         | Aleksei Stepanov                                                                                                 |
| bandit                    | 1.7.4       | Apache Software License                                                                             | PyCQA                                                                                                            |
| bcrypt                    | 3.2.0       | Apache Software License                                                                             | The Python Cryptographic Authority developers                                                                    |
| beautifulsoup4            | 4.11.1      | MIT License                                                                                         | Leonard Richardson                                                                                               |
| binaryornot               | 0.4.4       | BSD License                                                                                         | Audrey Roy Greenfeld                                                                                             |
| black                     | 22.3.0      | MIT License                                                                                         | Łukasz Langa                                                                                                     |
| bleach                    | 5.0.0       | Apache Software License                                                                             | UNKNOWN                                                                                                          |
| cachetools                | 4.2.4       | MIT License                                                                                         | Thomas Kemmer                                                                                                    |
| cachy                     | 0.3.0       | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| certifi                   | 2022.5.18.1 | Mozilla Public License 2.0 (MPL 2.0)                                                                | Kenneth Reitz                                                                                                    |
| cffi                      | 1.15.0      | MIT License                                                                                         | Armin Rigo, Maciej Fijalkowski                                                                                   |
| cfgv                      | 3.3.1       | MIT License                                                                                         | Anthony Sottile                                                                                                  |
| chardet                   | 4.0.0       | GNU Library or Lesser General Public License (LGPL)                                                 | Mark Pilgrim                                                                                                     |
| charset-normalizer        | 2.0.12      | MIT License                                                                                         | Ahmed TAHRI @Ousret                                                                                              |
| chemxor                   | 0.1.0       | Other/Proprietary License                                                                           | Ankur Kumar                                                                                                      |
| cleo                      | 0.8.1       | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| click                     | 8.1.3       | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| clikit                    | 0.6.2       | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| colorlog                  | 6.6.0       | MIT License                                                                                         | Sam Clements                                                                                                     |
| cookiecutter              | 1.7.3       | BSD License                                                                                         | Audrey Roy                                                                                                       |
| coverage                  | 6.4         | Apache Software License                                                                             | Ned Batchelder and 152 others                                                                                    |
| crashtest                 | 0.3.1       | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| cryptography              | 37.0.2      | Apache Software License; BSD License                                                                | The Python Cryptographic Authority and individual contributors                                                   |
| cycler                    | 0.11.0      | BSD License                                                                                         | Thomas A Caswell                                                                                                 |
| darglint                  | 1.8.1       | MIT License                                                                                         | terrencepreilly                                                                                                  |
| debugpy                   | 1.6.0       | Eclipse Public License 2.0 (EPL-2.0); MIT License                                                   | Microsoft Corporation                                                                                            |
| decorator                 | 5.1.1       | BSD License                                                                                         | Michele Simionato                                                                                                |
| defusedxml                | 0.7.1       | Python Software Foundation License                                                                  | Christian Heimes                                                                                                 |
| distlib                   | 0.3.4       | Python Software Foundation License                                                                  | Vinay Sajip                                                                                                      |
| dnspython                 | 2.2.1       | ISC                                                                                                 | Bob Halley                                                                                                       |
| dparse                    | 0.5.1       | MIT License                                                                                         | Jannis Gebauer                                                                                                   |
| dynaconf                  | 3.1.8       | MIT License                                                                                         | Bruno Rocha                                                                                                      |
| email-validator           | 1.2.1       | CC0 1.0 Universal (CC0 1.0) Public Domain Dedication                                                | Joshua Tauberer                                                                                                  |
| entrypoints               | 0.4         | MIT License                                                                                         | Thomas Kluyver                                                                                                   |
| eva                       | 1.0.1       | UNKNOWN                                                                                             | Microsoft Research EVA compiler team                                                                             |
| factory-boy               | 3.2.1       | MIT License                                                                                         | Mark Sandstrom                                                                                                   |
| fastapi                   | 0.66.1      | MIT License                                                                                         | Sebastián Ramírez                                                                                                |
| fastjsonschema            | 2.15.3      | BSD License                                                                                         | Michal Horejsek                                                                                                  |
| filelock                  | 3.7.0       | Public Domain                                                                                       | Benedikt Schmitt                                                                                                 |
| flake8                    | 4.0.1       | MIT License                                                                                         | Tarek Ziade                                                                                                      |
| flake8-annotations        | 2.9.0       | MIT License                                                                                         | S Co1                                                                                                            |
| flake8-bandit             | 3.0.0       | MIT License                                                                                         | Tyler Wince                                                                                                      |
| flake8-black              | 0.3.3       | MIT License                                                                                         | Peter J. A. Cock                                                                                                 |
| flake8-bugbear            | 22.4.25     | MIT License                                                                                         | Łukasz Langa                                                                                                     |
| flake8-docstrings         | 1.6.0       | MIT License                                                                                         | Simon ANDRÉ                                                                                                      |
| flake8-import-order       | 0.18.1      | GNU Lesser General Public License v3 (LGPLv3); MIT License                                          | Alex Stapleton                                                                                                   |
| flake8-polyfill           | 1.0.2       | MIT License                                                                                         | Ian Cordasco                                                                                                     |
| fonttools                 | 4.33.3      | MIT License                                                                                         | Just van Rossum                                                                                                  |
| forbiddenfruit            | 0.1.4       | GNU General Public License v3 or later (GPLv3+); MIT License                                        | Lincoln de Sousa                                                                                                 |
| frozenlist                | 1.3.0       | Apache Software License                                                                             | UNKNOWN                                                                                                          |
| fsspec                    | 2022.1.0    | BSD License                                                                                         | UNKNOWN                                                                                                          |
| gensim                    | 4.2.0       | LGPL-2.1-only                                                                                       | Radim Rehurek                                                                                                    |
| gevent                    | 21.8.0      | MIT License                                                                                         | Denis Bilenko                                                                                                    |
| gitdb                     | 4.0.9       | BSD License                                                                                         | Sebastian Thiel                                                                                                  |
| google-auth               | 2.6.6       | Apache Software License                                                                             | Google Cloud Platform                                                                                            |
| google-auth-oauthlib      | 0.4.6       | Apache Software License                                                                             | Google Cloud Platform                                                                                            |
| graphql-core              | 3.2.1       | MIT License                                                                                         | Christoph Zwerschke                                                                                              |
| greenlet                  | 1.1.2       | MIT License                                                                                         | Alexey Borzenkov                                                                                                 |
| griddify                  | 0.0.1       | MIT License                                                                                         | Miquel Duran-Frigola                                                                                             |
| grpcio                    | 1.46.3      | Apache Software License                                                                             | The gRPC Authors                                                                                                 |
| h11                       | 0.13.0      | MIT License                                                                                         | Nathaniel J. Smith                                                                                               |
| html5lib                  | 1.1         | MIT License                                                                                         | UNKNOWN                                                                                                          |
| httptools                 | 0.4.0       | MIT License                                                                                         | Yury Selivanov                                                                                                   |
| identify                  | 2.5.1       | MIT License                                                                                         | Chris Kuehl                                                                                                      |
| idna                      | 3.3         | BSD License                                                                                         | Kim Davies                                                                                                       |
| importlib-metadata        | 4.11.4      | Apache Software License                                                                             | Jason R. Coombs                                                                                                  |
| importlib-resources       | 5.7.1       | Apache Software License                                                                             | Barry Warsaw                                                                                                     |
| iniconfig                 | 1.1.1       | MIT License                                                                                         | Ronny Pfannschmidt, Holger Krekel                                                                                |
| ipykernel                 | 6.13.0      | BSD License                                                                                         | IPython Development Team                                                                                         |
| ipython                   | 7.34.0      | BSD License                                                                                         | The IPython Development Team                                                                                     |
| ipython-genutils          | 0.2.0       | BSD License                                                                                         | IPython Development Team                                                                                         |
| ipywidgets                | 7.7.0       | BSD License                                                                                         | IPython Development Team                                                                                         |
| itsdangerous              | 2.1.2       | BSD License                                                                                         | Armin Ronacher                                                                                                   |
| jedi                      | 0.18.1      | MIT License                                                                                         | David Halter                                                                                                     |
| jeepney                   | 0.8.0       | MIT License                                                                                         | Thomas Kluyver                                                                                                   |
| jinja2-time               | 0.2.0       | MIT License                                                                                         | Raphael Pierzina                                                                                                 |
| jmespath                  | 0.10.0      | MIT License                                                                                         | James Saryerwinnie                                                                                               |
| joblib                    | 1.1.0       | BSD License                                                                                         | Gael Varoquaux                                                                                                   |
| json5                     | 0.9.8       | Apache Software License                                                                             | Dirk Pranke                                                                                                      |
| jsonschema                | 4.5.1       | MIT License                                                                                         | Julian Berman                                                                                                    |
| jupyter-client            | 7.3.1       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| jupyter-core              | 4.10.0      | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| jupyter-server            | 1.17.0      | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| jupyterlab                | 3.4.2       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| jupyterlab-pygments       | 0.2.2       | BSD                                                                                                 | Jupyter Development Team                                                                                         |
| jupyterlab-server         | 2.14.0      | BSD License                                                                                         | UNKNOWN                                                                                                          |
| jupyterlab-widgets        | 1.1.0       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| kedro                     | 0.18.1      | Apache Software License (Apache 2.0)                                                                | Kedro                                                                                                            |
| kedro-viz                 | 4.6.0       | Apache Software License (Apache 2.0)                                                                | Kedro                                                                                                            |
| keyring                   | 23.5.0      | MIT License; Python Software Foundation License                                                     | Kang Zhang                                                                                                       |
| kiwisolver                | 1.4.2       | BSD License                                                                                         | UNKNOWN                                                                                                          |
| lap                       | 0.4.0       | BSD (2-clause)                                                                                      | UNKNOWN                                                                                                          |
| lark-parser               | 0.12.0      | MIT License                                                                                         | Erez Shinan                                                                                                      |
| llvmlite                  | 0.38.1      | BSD                                                                                                 | UNKNOWN                                                                                                          |
| lockfile                  | 0.12.2      | MIT License                                                                                         | OpenStack                                                                                                        |
| loguru                    | 0.5.3       | MIT License                                                                                         | Delgan                                                                                                           |
| logzero                   | 1.7.0       | MIT License                                                                                         | Chris Hager                                                                                                      |
| matplotlib                | 3.5.2       | Python Software Foundation License                                                                  | John D. Hunter, Michael Droettboom                                                                               |
| matplotlib-inline         | 0.1.3       | BSD 3-Clause                                                                                        | IPython Development Team                                                                                         |
| mccabe                    | 0.6.1       | MIT License                                                                                         | Ian Cordasco                                                                                                     |
| mistune                   | 0.8.4       | BSD License                                                                                         | Hsiaoming Yang                                                                                                   |
| mpmath                    | 1.2.1       | BSD License                                                                                         | Fredrik Johansson                                                                                                |
| msgpack                   | 1.0.3       | Apache Software License                                                                             | Inada Naoki                                                                                                      |
| multidict                 | 6.0.2       | Apache Software License                                                                             | Andrew Svetlov                                                                                                   |
| mypy                      | 0.942       | MIT License                                                                                         | Jukka Lehtosalo                                                                                                  |
| mypy-extensions           | 0.4.3       | MIT License                                                                                         | The mypy developers                                                                                              |
| names                     | 0.3.0       | MIT License                                                                                         | Trey Hunner                                                                                                      |
| nbclassic                 | 0.3.7       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| nbclient                  | 0.6.3       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| nbconvert                 | 6.5.0       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| nbformat                  | 5.4.0       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| nest-asyncio              | 1.5.5       | BSD License                                                                                         | Ewald R. de Wit                                                                                                  |
| networkx                  | 2.8.2       | BSD License                                                                                         | Aric Hagberg                                                                                                     |
| nglview                   | 3.0.3       | MIT License                                                                                         | Alexander S. Rose, Hai Nguyen                                                                                    |
| nltk                      | 3.7         | Apache Software License                                                                             | NLTK Team                                                                                                        |
| nodeenv                   | 1.6.0       | BSD License                                                                                         | Eugene Kalinin                                                                                                   |
| notebook                  | 6.4.11      | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| notebook-shim             | 0.1.0       | BSD License                                                                                         | Jupyter Development Team                                                                                         |
| nox                       | 2022.1.7    | Apache Software License                                                                             | Alethea Katherine Flowers                                                                                        |
| numba                     | 0.55.2      | BSD License                                                                                         | UNKNOWN                                                                                                          |
| numpy                     | 1.22.4      | BSD License                                                                                         | Travis E. Oliphant et al.                                                                                        |
| oauthlib                  | 3.2.0       | BSD License                                                                                         | The OAuthlib Community                                                                                           |
| onnx                      | 1.11.0      | Apache License v2.0                                                                                 | ONNX                                                                                                             |
| packaging                 | 21.3        | Apache Software License; BSD License                                                                | Donald Stufft and individual contributors                                                                        |
| pandas                    | 1.3.4       | BSD License                                                                                         | The Pandas Development Team                                                                                      |
| pandocfilters             | 1.5.0       | BSD License                                                                                         | John MacFarlane                                                                                                  |
| parso                     | 0.8.3       | MIT License                                                                                         | David Halter                                                                                                     |
| pastel                    | 0.2.1       | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| pathspec                  | 0.9.0       | Mozilla Public License 2.0 (MPL 2.0)                                                                | Caleb P. Burns                                                                                                   |
| pbr                       | 5.9.0       | Apache Software License                                                                             | OpenStack                                                                                                        |
| pep517                    | 0.12.0      | MIT License                                                                                         | Thomas Kluyver                                                                                                   |
| pexpect                   | 4.8.0       | ISC License (ISCL)                                                                                  | Noah Spurrier; Thomas Kluyver; Jeff Quast                                                                        |
| pickleshare               | 0.7.5       | MIT License                                                                                         | Ville Vainio                                                                                                     |
| pip-tools                 | 6.6.2       | BSD License                                                                                         | Vincent Driessen                                                                                                 |
| pkginfo                   | 1.8.2       | MIT License                                                                                         | Tres Seaver, Agendaless Consulting                                                                               |
| platformdirs              | 2.5.2       | MIT License                                                                                         | UNKNOWN                                                                                                          |
| plotly                    | 5.8.0       | MIT                                                                                                 | Chris P                                                                                                          |
| pluggy                    | 1.0.0       | MIT License                                                                                         | Holger Krekel                                                                                                    |
| poetry                    | 1.1.13      | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| poetry-core               | 1.0.8       | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| poyo                      | 0.5.0       | MIT License                                                                                         | Raphael Pierzina                                                                                                 |
| pre-commit                | 2.19.0      | MIT License                                                                                         | Anthony Sottile                                                                                                  |
| prometheus-client         | 0.14.1      | Apache Software License                                                                             | Brian Brazil                                                                                                     |
| prompt-toolkit            | 3.0.29      | BSD License                                                                                         | Jonathan Slenders                                                                                                |
| protobuf                  | 3.19.1      | 3-Clause BSD License                                                                                | UNKNOWN                                                                                                          |
| psutil                    | 5.9.1       | BSD License                                                                                         | Giampaolo Rodola                                                                                                 |
| ptyprocess                | 0.7.0       | ISC License (ISCL)                                                                                  | Thomas Kluyver                                                                                                   |
| py                        | 1.11.0      | MIT License                                                                                         | holger krekel, Ronny Pfannschmidt, Benjamin Peterson and others                                                  |
| pyDeprecate               | 0.3.2       | MIT                                                                                                 | Jiri Borovec                                                                                                     |
| pyarrow                   | 6.0.0       | Apache Software License                                                                             | UNKNOWN                                                                                                          |
| pyasn1                    | 0.4.8       | BSD License                                                                                         | Ilya Etingof                                                                                                     |
| pyasn1-modules            | 0.2.8       | BSD License                                                                                         | Ilya Etingof                                                                                                     |
| pycodestyle               | 2.8.0       | MIT License                                                                                         | Johann C. Rocholl                                                                                                |
| pycparser                 | 2.21        | BSD License                                                                                         | Eli Bendersky                                                                                                    |
| pydantic                  | 1.9.1       | MIT License                                                                                         | Samuel Colvin                                                                                                    |
| pydocstyle                | 6.1.1       | MIT License                                                                                         | Amir Rachum                                                                                                      |
| pyflakes                  | 2.4.0       | MIT License                                                                                         | A lot of people                                                                                                  |
| pylev                     | 1.4.0       | BSD License                                                                                         | Daniel Lindsley                                                                                                  |
| pymbolic                  | 2021.1      | MIT License                                                                                         | Andreas Kloeckner                                                                                                |
| pynndescent               | 0.5.7       | BSD                                                                                                 | Leland McInnes                                                                                                   |
| pyparsing                 | 3.0.9       | MIT License                                                                                         | UNKNOWN                                                                                                          |
| pyrsistent                | 0.18.1      | MIT License                                                                                         | Tobias Gustafsson                                                                                                |
| pytest                    | 7.1.2       | MIT License                                                                                         | Holger Krekel, Bruno Oliveira, Ronny Pfannschmidt, Floris Bruynooghe, Brianna Laugher, Florian Bruhin and others |
| pytest-cov                | 3.0.0       | MIT License                                                                                         | Marc Schlaich                                                                                                    |
| pytest-mock               | 3.7.0       | MIT License                                                                                         | Bruno Oliveira                                                                                                   |
| python-dateutil           | 2.8.2       | Apache Software License; BSD License                                                                | Gustavo Niemeyer                                                                                                 |
| python-dotenv             | 0.20.0      | BSD License                                                                                         | Saurabh Kumar                                                                                                    |
| python-json-logger        | 2.0.2       | BSD License                                                                                         | Zakaria Zajac                                                                                                    |
| python-multipart          | 0.0.5       | Apache Software License                                                                             | Andrew Dunham                                                                                                    |
| python-slugify            | 6.1.2       | MIT License                                                                                         | Val Neekman                                                                                                      |
| pytools                   | 2022.1.7    | MIT License                                                                                         | Andreas Kloeckner                                                                                                |
| pytorch-lightning         | 1.6.3       | Apache Software License                                                                             | William Falcon et al.                                                                                            |
| pytz                      | 2022.1      | MIT License                                                                                         | Stuart Bishop                                                                                                    |
| pyzmq                     | 23.0.0      | BSD License; GNU Library or Lesser General Public License (LGPL)                                    | Brian E. Granger, Min Ragan-Kelley                                                                               |
| rdkit-pypi                | 2022.3.2.1  | BSD-3-Clause                                                                                        | Christopher Kuenneth                                                                                             |
| regex                     | 2022.4.24   | Apache Software License                                                                             | Matthew Barnett                                                                                                  |
| requests                  | 2.27.1      | Apache Software License                                                                             | Kenneth Reitz                                                                                                    |
| requests-oauthlib         | 1.3.1       | BSD License                                                                                         | Kenneth Reitz                                                                                                    |
| requests-toolbelt         | 0.9.1       | Apache Software License                                                                             | Ian Cordasco, Cory Benfield                                                                                      |
| rope                      | 0.21.1      | GNU Lesser General Public License v3 or later (LGPLv3+)                                             | Ali Gholami Rudi                                                                                                 |
| rsa                       | 4.8         | Apache Software License                                                                             | Sybren A. Stüvel                                                                                                 |
| safety                    | 1.10.3      | MIT License                                                                                         | pyup.io                                                                                                          |
| scikit-learn              | 1.1.1       | new BSD                                                                                             | UNKNOWN                                                                                                          |
| scipy                     | 1.7.3       | BSD License                                                                                         | UNKNOWN                                                                                                          |
| semver                    | 2.13.0      | BSD License                                                                                         | Kostiantyn Rybnikov                                                                                              |
| setuptools-scm            | 6.4.2       | MIT License                                                                                         | Ronny Pfannschmidt                                                                                               |
| shellingham               | 1.4.0       | ISC License (ISCL)                                                                                  | Tzu-ping Chung                                                                                                   |
| six                       | 1.16.0      | MIT License                                                                                         | Benjamin Peterson                                                                                                |
| smart-open                | 6.0.0       | MIT License                                                                                         | Radim Rehurek                                                                                                    |
| smmap                     | 5.0.0       | BSD License                                                                                         | Sebastian Thiel                                                                                                  |
| sniffio                   | 1.2.0       | Apache Software License; MIT License                                                                | Nathaniel J. Smith                                                                                               |
| snowballstemmer           | 2.2.0       | BSD License                                                                                         | Snowball Developers                                                                                              |
| soupsieve                 | 2.3.2.post1 | MIT License                                                                                         | UNKNOWN                                                                                                          |
| starlette                 | 0.14.2      | BSD License                                                                                         | Tom Christie                                                                                                     |
| stevedore                 | 3.5.0       | Apache Software License                                                                             | OpenStack                                                                                                        |
| strawberry-graphql        | 0.114.0     | MIT License                                                                                         | Patrick Arminio                                                                                                  |
| syft                      | 0.6.0       | Apache-2.0                                                                                          | OpenMined                                                                                                        |
| sympy                     | 1.9         | BSD License                                                                                         | SymPy development team                                                                                           |
| tenacity                  | 8.0.1       | Apache Software License                                                                             | Julien Danjou                                                                                                    |
| tenseal                   | 0.3.11      | Apache-2.0                                                                                          | OpenMined                                                                                                        |
| tensorboard               | 2.9.0       | Apache Software License                                                                             | Google Inc.                                                                                                      |
| tensorboard-data-server   | 0.6.1       | Apache Software License                                                                             | Google Inc.                                                                                                      |
| tensorboard-plugin-wit    | 1.8.1       | Apache 2.0                                                                                          | Google Inc.                                                                                                      |
| terminado                 | 0.15.0      | BSD License                                                                                         | UNKNOWN                                                                                                          |
| text-unidecode            | 1.3         | Artistic License; GNU General Public License (GPL); GNU General Public License v2 or later (GPLv2+) | Mikhail Korobov                                                                                                  |
| threadpoolctl             | 3.1.0       | BSD License                                                                                         | Thomas Moreau                                                                                                    |
| tinycss2                  | 1.1.1       | BSD License                                                                                         | UNKNOWN                                                                                                          |
| toml                      | 0.10.2      | MIT License                                                                                         | William Pearson                                                                                                  |
| tomli                     | 2.0.1       | MIT License                                                                                         | UNKNOWN                                                                                                          |
| tomlkit                   | 0.10.2      | MIT License                                                                                         | Sébastien Eustace                                                                                                |
| toposort                  | 1.7         | Apache Software License                                                                             | "Eric V. Smith"                                                                                                  |
| torch                     | 1.11.0      | BSD License                                                                                         | PyTorch Team                                                                                                     |
| torch-tb-profiler         | 0.4.0       | BSD License                                                                                         | PyTorch Team                                                                                                     |
| torchani                  | 2.2         | MIT                                                                                                 | Xiang Gao                                                                                                        |
| torchmetrics              | 0.8.2       | Apache Software License                                                                             | PyTorchLightning et al.                                                                                          |
| torchvision               | 0.12.0      | BSD                                                                                                 | PyTorch Core Team                                                                                                |
| tornado                   | 6.1         | Apache Software License                                                                             | Facebook                                                                                                         |
| tqdm                      | 4.64.0      | MIT License; Mozilla Public License 2.0 (MPL 2.0)                                                   | UNKNOWN                                                                                                          |
| traitlets                 | 5.2.1.post0 | BSD License                                                                                         | UNKNOWN                                                                                                          |
| typeguard                 | 2.13.3      | MIT License                                                                                         | Alex Grönholm                                                                                                    |
| typing-extensions         | 4.2.0       | Python Software Foundation License                                                                  | UNKNOWN                                                                                                          |
| umap-learn                | 0.5.3       | BSD                                                                                                 | UNKNOWN                                                                                                          |
| urllib3                   | 1.26.9      | MIT License                                                                                         | Andrey Petrov                                                                                                    |
| uvicorn                   | 0.17.6      | BSD License                                                                                         | Tom Christie                                                                                                     |
| uvloop                    | 0.16.0      | Apache Software License; MIT License                                                                | Yury Selivanov                                                                                                   |
| virtualenv                | 20.14.1     | MIT License                                                                                         | Bernat Gabor                                                                                                     |
| watchgod                  | 0.8.2       | MIT License                                                                                         | Samuel Colvin                                                                                                    |
| wcwidth                   | 0.2.5       | MIT License                                                                                         | Jeff Quast                                                                                                       |
| webencodings              | 0.5.1       | BSD License                                                                                         | Geoffrey Sneddon                                                                                                 |
| websocket-client          | 1.3.2       | Apache Software License                                                                             | liris                                                                                                            |
| websockets                | 10.3        | BSD License                                                                                         | Aymeric Augustin                                                                                                 |
| widgetsnbextension        | 3.6.0       | BSD License                                                                                         | IPython Development Team                                                                                         |
| yarl                      | 1.7.2       | Apache Software License                                                                             | Andrew Svetlov                                                                                                   |
| zipp                      | 3.8.0       | MIT License                                                                                         | Jason R. Coombs                                                                                                  |
| zope.event                | 4.5.0       | Zope Public License                                                                                 | Zope Foundation and Contributors                                                                                 |
| zope.interface            | 5.4.0       | Zope Public License                                                                                 | Zope Foundation and Contributors                                                                                 |
