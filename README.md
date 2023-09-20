# NoiseCut

![NoiseCut logo](docs/artwork/NoiseCut_logo.jpg)

`noisecut` is an easy-to-use Python package for the implementation of
tree-structured functional networks (FNs) as a model class for the
classification of binary data with prior knowledge on input features.
FNs can be viewed as modular neural networks, where the structure of
the links between the modules and the information flow from input variables
to the output variable is pre-determined. Here, each module of the FN is
simply represented as a black-box module. The identification of an FN, i.e.,
learning the input-output function of the FN, is then decomposed to the
identification of the individual interior black-box modules.

`noisecut` can be used for any tree-structured FNs which has the below
criteria. It should have
1. two hidden layer,
2. arbitrary number of black-box modules in the first hidden layer,
3. only one black-box module in the second hidden layer,
4. each input feature goes only to one black-box module (tree structure).

## Installation

### Dependencies

- Python (>=3.9)
- numpy
- pandas
- scipy
- gurobipy

### User installation

Bofore you can use NoiseCut package, you need to install `noisecut` using pip:

```bash
$ pip install noisecut
```

## Usage

Use cases of the useful functions of `noisecut` package has been provided on
examples section. Examples show you how to use the package to fit the model and
investigate the predicted results in score, probability or simple binary
output format.

## License

`noisecut` was created by Hedieh Mirzaieazar. It is licensed under the terms
of the GPLv3 license.
