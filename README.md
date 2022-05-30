# ChemXor

Privacy Preserving AI/ML for Drug Discovery

---

## Overview

ChemXor is an open source library for training and evaluating PyTorch models on FHE(Fully homormorphic encryption) encrypted inputs with no manual code changes to the original model. It also provides convenience functions to quickly query and serve these models as a service with strong privacy guarantees for the end user. It is built on top of TenSEAL, Pytorch and ONNX.

### What is Fully Homomorphic Encryption (FHE)?

### Why do you need FHE for Machine Learning?

## Getting Started

Install the `chemxor` library from PyPi.

```bash
pip install chemxor
```

Use the `FHECryptor` class to convert Pytorch models for FHE inputs.

```python
import tenseal as ts
from chemxor.crypt import FHECryptor
from chemxor.models.cryptic_sage import CrypticSage

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
converted_model = fhe_cryptor.convert_model(model)

# Converted model is still a Pytorch lightning module
# So use it as usual for evaluating encrypted inputs
enc_output = converted_model(enc_input)
```

Quickly serve models as a service

```python

from chemxor.service import create_model_server

# `create_model_server` returns a flask app
flask_app = create_model_server(model)

if __name__ == "__main__":
    flask_app.run()
```

Query models

```bash
chemxor query -i [input file path] [model url]
```

## Developing

We use poetry to manage project dependecies. Use poetry to install project in editable mode.

```bash
poetry install
```

## License

This project is licensed under GNU AFFERO GENERAL PUBLIC LICENSE Version 3.
