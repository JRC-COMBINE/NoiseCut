# Contributing guidelines

- [Contributing guidelines](#contributing-guidelines)
  - [Introduction](#introduction)
  - [Writing helpful bug reports](#writing-helpful-bug-reports)
  - [Installing the latest version](#installing-the-latest-version)
  - [Setting up a local development environment](#setting-up-a-local-development-environment)
    - [Fork the repository](#fork-the-repository)
    - [Installing a proper python version](#installing-a-proper-python-version)
    - [Creating a python environment](#creating-a-python-environment)
    - [Installing from source](#installing-from-source)
    - [Code checks with precommit](#code-checks-with-precommit)
    - [Unit tests with pytest](#unit-tests-with-pytest)
    - [Code Coverage](#code-coverage)
  - [Pull Requests (PRs)](#pull-requests-prs)
    - [Etiquette for creating PRs](#etiquette-for-creating-prs)
    - [Checklist for publishing PRs](#checklist-for-publishing-prs)

## Introduction
Thank you for contributing to NoiseCut. NoiseCut is an open source collective effort,
and contributions of all forms are welcome!

You can contribute by:

- Submitting bug reports and features requests on the GitHub [issue
  tracker][issues],
- Contributing fixes and improvements via [Pull Requests][pulls], or
- Discussing ideas and questions in the [Discussions forum][discussions].

[issues]: https://github.com/JRC-COMBINE/NoiseCut/issues
[pulls]: https://github.com/JRC-COMBINE/NoiseCut/pulls
[discussions]: https://github.com/JRC-COMBINE/NoiseCut/discussions

## Writing helpful bug reports

When submitting bug reports on the [issue tracker][issues], it is very helpful
for the maintainers to include a good **Minimal Reproducible Example** (MRE).

An MRE should be:

- **Minimal**: Use as little code as possible that still produces the same
  problem.
- **Self-contained**: Include everything needed to reproduce your problem,
  including imports and input data.
- **Reproducible**: Test the code you're about to provide to make sure it
  reproduces the problem.

For more information, see [How To Craft Minimal Bug
Reports](https://matthewrocklin.com/minimal-bug-reports).

## Installing the latest version

To get the very latest version of NoiseCut, you can pip-install the library directly
from the `main` branch:

```bash
pip install git+https://github.com/JRC-COMBINE/NoiseCut.git@main
```

This can be useful to test if a particular issue or bug has been fixed since the
most recent release.

Alternatively, if you are considering making changes to the code you can clone
the repository and install your local copy as described below.

## Setting up a local development environment

### Fork the repository

Click [this link](https://github.com/JRC-COMBINE/NoiseCut/fork) to fork the 
repository on GitHub to your user area.

Clone the repository to your local environment, using the URL provided by the
green `<> Code` button on your projects home page.

### Installing a proper python version

Install a version of python in your local machine. Then, Create a specific version of python, e.g. with pyenv:
in mac:
```bash
brew update
brew install pyenv
pyenv install 3.10.12
```
See the installed version of python with:
```bash
pyenv versions
```
To set the specific python version as the default python for your entire system, use the below command:
```bash
pyenv global 3.10.12
```
To set the specific python version just for a shell:
```bash
pyenv shell 3.10.12
```
Check the python version of the shell or system by:
```bash
python --version
```

### Creating a python environment

Create a new isolated environment for the project, e.g. with venv:

```bash
python -m venv ./venv
. venv/bin/activate
```

### Installing from source

Pip-install the project, in the path in which [pyproject.toml](pyproject.toml) 
file exists, with the `--editable` flag, which ensures that any
changes you make to the source code are immediately reflected in your
environment.

```bash
pip install --editable '.[dev,notebook]'
```

The various pip extras are defined in [pyproject.toml](pyproject.toml):

- `test`: a minimal set of dependencies to run pytest.
- `release`: dependencies for publishing package on pypi.
- `static-code-qa`: dependencies for linting the package.
- `notebook`: required dependencies for running jupyter notebooks.
- `docs`: dependencies for building the docs with Sphinx.
- `dev`: a combination of `test`, `release`, `static-code-qa`, and `docs` dependencies

### Code checks with precommit

We use [pre-commit hooks](https://pre-commit.com/#install) to run code checks. When the package is installed with `dev` dependencies, `pre-commit` should be installed using pip in the virtual environment.

To run the checks on all files, use:

```bash
pre-commit install
pre-commit run --all-files
```
To run on a specific file, use:
```bash
pre-commit run --files src/noisecut/model/*.py
```

To skip a specific hook, e.g. isort, use:
```bash
SKIP=isort pre-commit run --all-files
```

[Ruff](https://beta.ruff.rs/docs/) is used as a linter, and it is enabled as a
pre-commit hook. You can also run `ruff` locally with:

```bash
pip install ruff
ruff check .
```

### Unit tests with pytest

The unit tests can be run locally with:

```bash
./run.sh test
```

or:

```bash
pytest
```

To skip slow tests, use:

```bash
./run.sh test:quick
```

### Code Coverage

The coverage of unit tests can be seen as a web report with:

```bash
./run.sh serve-coverage-report
```

You can see the result in your browser by going to the [localhost:8000](http://localhost:8000).

## Pull Requests (PRs)

### Etiquette for creating PRs

Before starting on a PR, please make a proposal by **opening an Issue**, checking for any duplicates. This isn't necessary for trivial PRs such as fixing a typo.

**Keep the scope small**. This makes PRs a lot easier to review. Separate functional code changes (such as bug fixes) from refactoring changes (such as style improvements). PRs should contain one or the other, but not both.

Open a **Draft PR** as early as possible, do not wait until the feature is ready. Work on a feature branch with a descriptive name such as `fix/name-of-error` or `doc/contributing`.

Try to use a descriptive title.

### Checklist for publishing PRs

Before marking your PR as "ready for review" (by removing the `Draft` status),
please ensure:

- Your feature branch is up-to-date with the master branch,
- All [pre-commit hooks](#code-checks-with-precommit) pass,
- [Unit tests](#unit-tests-with-pytest) pass, and
- Unit tests have been added (if your PR adds any new features or fixes a bug). [Code coverage](#code-coverage) is above 90% for your code.
