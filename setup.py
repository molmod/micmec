import numpy as np
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import build_ext, cythonize


external = Extension(
    "micmec.pes.ext",
    sources=["micmec/pes/ext.pyx", "micmec/pes/domain.c"],
    depends=["micmec/pes/domain.h", "micmec/pes/domain.pxd"],
    include_dirs=[np.get_include()]
)

setup(
    name="micmec",
    version="1.0",
    description="The first implementation of the micromechanical model, ever.",
    author="Joachim Vandewalle",
    author_email="joachim.vandewalle@hotmail.be",
    url="https://github.com/Jlvdwall/micmec",
    package_dir={"micmec": "micmec"},
    packages=["micmec", "micmec/pes", "micmec/sampling", "micmec/analysis"],
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
    zip_safe=False,
    setup_requires=[
        "numpy>=1.5",
        "cython>=0.26"
    ],
    install_requires=[
        "numpy>=1.5",
        "cython>=0.26",
        "matplotlib>1.0.0",
        "h5py>=2.0.0",
        "molmod>=1.4.1"
    ],
    ext_modules=[external],
    classifiers=[
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": ["micmec_builder=micmec.builder.builder:main"]
    }
)

