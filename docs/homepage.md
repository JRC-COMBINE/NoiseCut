# NoiseCut

![NoiseCut logo](artwork/NoiseCut_logo.jpg)

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
- cplex
- docplex

### User installation

Bofore you can use NoiseCut package, you need to install `noisecut` using pip:

```bash
$ pip install noisecut
```

## Simple demo

Code snippet shown below summarizes a complete workflow, starting with
the generation of synthetic data, proceeding to the division of data into
training and testing sets, and concluding with model fitting and result
evaluation.

```python
from noisecut.model.noisecut_coder import Metric
from noisecut.model.noisecut_model import NoiseCut
from noisecut.tree_structured.data_manipulator import DataManipulator
from noisecut.tree_structured.sample_generator import SampleGenerator

# Synthetic data generation
gen_dataset = SampleGenerator(
    [4, 4, 4], allowance_rand=True
)  # [4,4,4] determines the number of inputs to each black box of the FN model
X, y = gen_dataset.get_complete_data_set()

# Add noise in data labeling. Train and test set split.
x_noisy, y_noisy = DataManipulator().get_noisy_data(X, y, percentage_noise=10)
x_train, y_train, x_test, y_test = DataManipulator().split_data(
    x_noisy, y_noisy, percentage_training_data=50
)

# Training
mdl = NoiseCut(
    n_input_each_box=[4, 4, 4]
)  # 'n_input_each_box' should fit to the generated data
mdl.fit(x_train, y_train)

# Evaluation
y_pred = mdl.predict(x_test)
accuracy, recall, precision, F1 = Metric.set_confusion_matrix(y_test, y_pred)
```

## Usage

Various use cases of the useful functions of `noisecut` package are provided
as jupyter notebooks:

- {doc}`notebooks/Usage_example_of_NoiseCut`
- {doc}`notebooks/Generation_of_synthetic_data`
- {doc}`notebooks/Noise-tolerant_classification`
- {doc}`notebooks/Classification_with_reduced_training_data`

Examples show how to use the package to fit the model and investigate
the predicted results in score, probability or simple binary output format.

## License

`noisecut` was created by Hedieh Mirzaieazar and Moein E. Samadi. It is
licensed under the terms of the GPLv3 license.
