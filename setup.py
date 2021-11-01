from setuptools import find_packages, setup

requirements = (
    "gpflow>=2.1.3",
    "pytest",
)

extra_requirements = {
    "demos": (
        "matplotlib",
        "tqdm",
        "sklearn",
        'jupytex',
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
