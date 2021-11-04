from setuptools import find_packages, setup

requirements = (
    "tensorflow==2.6.0",
    "tensorflow-probability==0.14.1",
    "gpflow",
    "pytest",
)

extra_requirements = {
    "demos": (
        "matplotlib",
        "tqdm",
        "sklearn",
    ),
}

setup(
    name="t-SVGP",
    version="0",
    license="Creative Commons Attribution-Noncommercial-Share Alike license",
    packages=find_packages(exclude=["demos*"]),
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require=extra_requirements,
)
