[![Quality checks and Tests](https://github.com/AaltoML/t-SVGP/actions/workflows/quality-check.yaml/badge.svg)](https://github.com/AaltoML/t-SVGP/actions/workflows/quality-check.yaml)

# Dual Parameterization of Sparse Variational Gaussian Processes

This repository is the official implementation of the methods in the publication:

* V. Adam, P.E. Chang, M.E. Khan and A. Solin (2021). **Dual Parameterization of Sparse Variational Gaussian Processes**. *To appear at Advances in Neural Information Processing Systems (NeurIPS)*. [[arXiv]](https://arxiv.org/abs/XXXX.XXXX)


The paper's main result shows that an alternative (dual) parameterization for SVGP models leads to a better objective for learning and allows for faster inference via natural gradient descent.

## Supplemental material
Structure of the supplemental material folder:

* `scr` contains the source code
* `demos` contains scripts to reproduce some of the experiments presented in the paper  
* `notebooks` contains a Jupyter notebook in Python illustrating the proposed approach
* `tests` contains unit and integration tests for the source code

## Installation

We recommend using Python version 3.7.3 and pip version 20.1.1.
To install the package, run:

```
pip install -e .
```
To install the dependencies needed to run `demos`, use `pip install -e .[demos]`.

### Notebooks

To build the notebooks from source, run
```
jupytext --to notebook [filename].py
```


## Citation
If you use the code in this repository for your research, please cite the paper as follows:
```bibtex
@inproceedings{adam2021,
  title={Dual Parameterization of Sparse Variational Gaussian Processes},
  author={Adam, Vincent and Chang, Paul Edmund and Khan, Mohammad Emtiyaz and Solin, Arno},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## Contributing

For all correspondence, please contact vincenta@gatsby.ucl.ac.uk.

## License

This software is provided under the [MIT license](LICENSE).






