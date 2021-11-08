# Dual Parameterization of Sparse Variational Gaussian Processes

[![Quality checks and Tests](https://github.com/AaltoML/t-SVGP/actions/workflows/quality-check.yaml/badge.svg)](https://github.com/AaltoML/t-SVGP/actions/workflows/quality-check.yaml)
[![Docs build](https://github.com/AaltoML/t-SVGP/actions/workflows/deploy_notebooks.yaml/badge.svg)](https://github.com/AaltoML/t-SVGP/actions/workflows/deploy_notebooks.yaml)

[Documentation](https://aaltoml.github.io/t-SVGP/) |
[Notebooks](https://aaltoml.github.io/t-SVGP/notebooks.html) |
[API reference](https://aaltoml.github.io/t-SVGP/autoapi/src/index.html)

## Introduction

This repository is the official implementation of the methods in the publication:

* V. Adam, P.E. Chang, M.E. Khan, and A. Solin (2021). **Dual Parameterization of Sparse Variational Gaussian Processes**. In *Advances in Neural Information Processing Systems (NeurIPS)*. [[arXiv]](https://arxiv.org/abs/XXXX.XXXX)

The paper's main result shows that an alternative (dual) parameterization for SVGP models leads to a better objective for learning and allows for faster inference via natural gradient descent.

## Repository structure

The repository has the following folder structure:

* `scr` contains the source code
* `experiments` contains scripts to reproduce some of the experiments presented in the paper  
* `docs` contains documentation in the form of notebooks and an api reference.
* `tests` contains unit and integration tests for the source code

## Installation

We recommend using Python version 3.7.3 and pip version 20.1.1.
To install the package, run:

```bash
pip install -e .
```

To run the tests, notebooks, build the docs or run the experiments, install the dependencies:

```bash
pip install \
  -r tests_requirements.txt \
  -r notebook_requirements.txt \
  -r docs/docs_requirements.txt \
  -e .
```


### Notebooks

To build the notebooks from source, use jupytext:
```bash
jupytext --to notebook [filename].py
```

## Citation
If you use the code in this repository for your research, please cite the paper as follows:
```bibtex
@inproceedings{adam2021dual,
  title={Dual Parameterization of Sparse Variational {G}aussian Processes},
  author={Adam, Vincent and Chang, Paul Edmund and Khan, Mohammad Emtiyaz and Solin, Arno},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Contributing

For all correspondence, please contact [vincenta@gatsby.ucl.ac.uk](mailto:vincenta@gatsby.ucl.ac.uk).

## License

This software is provided under the [MIT license](LICENSE).






