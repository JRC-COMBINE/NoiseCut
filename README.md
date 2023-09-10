# NoiseCut

A Tree-structured Hybrid model has been implemented in this package. It can be used to fit a hybrid model which has the below criteria.
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

It is better to install NoiseCut Package in a virtual environment.
Open a terminal and go to the directory where the package is cloned. Write the below command:
```bash
$ python -m venv ./venv
$ . venv/bin/activate
```
Now, the virtual environment should be activated. Then, the package can be installed with pip in editable mode by typing the below command:

```bash
$ pip install -e '.[dev, docs]'
```

## Usage

Use cases of the useful functions of this package has been provided on the [example.ipynb](docs/example.ipynb)
notebook. It shows how to use the package to fit the model and investigate the predicted results in score,
probability or simple binary output format.

## License

`noisecut` was created by Hedieh Mirzaieazar. It is licensed under the terms of the GPLv3 license.
