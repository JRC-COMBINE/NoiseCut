# NoiseCut

![NoiseCut logo](docs/artwork/NoiseCut_logo.jpg)

A Tree-structured Hybrid model has been implemented in this package. It can be 
used to fit a hybrid model which has the below criteria.
1. two hidden layer
2. arbitrary number of black boxes in the first hidden layer
3. only one black box in the second hidden layer
4. each input feature goes only to one black box.

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
