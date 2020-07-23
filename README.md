# Cerbero

*Multi-task learning made easy*

Cerbero aims to pick up the multi-task learning framework that was started by Snorkel before they dropped support.
Rather than focusing on weak supervision and labeling functions Cerbero will mainly focus on allowing users to build and train multi-task models with minimal fuss.

## Quickstart

Cerbero requires Python 3.6 or later. To install Cerbero, clone the environment and pip install

```bash
git clone git@github.com:strongio/cerbero.git
cd cerbero
pip install -e .
```

### Examples

There are a couple of examples which go over basic usage of Cerbero and the multi-task framework:

- [Basic Example](examples/basic_example): simple example showcasing multi-task learning with Cerbero on synthetic data
- [CIFAR-10 Example](examples/cifar10_example): A classification/regression multi-task example using the CIFAR-10 dataset

## Development Environment

Following it's predecessor Snorkel, Cerbero uses [tox](https://tox.readthedocs.io) to manage development environments
To get started, [install tox](https://tox.readthedocs.io/en/latest/install.html), clone Cerbero, then use `tox` to create a development environment.

```bash
git clone git@github.com:strongio/cerbero.git
pip3 install -U tox
cd cerbero
tox --devenv .env
```

Running `tox --devenv .env` will install create a virtual environment with Cerbero
and all of its dependencies installed in the directory `.env`.
This can be used in a number of ways, e.g. with `source .env/bin/activate`.

### Testing

At the moment there are just a few tests which we can run using the following `tox` commands:

```bash
tox -e py37  # Run unit tests pytest in Python 3.7
tox -e check # Check style with black
tox -e type # Run static type checking with mypy
tox -e fix # Fix style issues with black
```
