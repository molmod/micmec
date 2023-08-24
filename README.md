![THE MICROMECHANICAL PROCEDURE](https://github.com/Jlvdwall/micmec/blob/main/docs/figs/micmec_proc_wide.png)

MicMec is a geometry optimization and molecular dynamics code for coarse-grained crystalline materials.
It is the first implementation of the micromechanical model, ever.
The theoretical groundwork of the micromechanical model was originally established in:

> S. M. J. Rogge, “The micromechanical model to computationally investigate cooperative and correlated phenomena in metal-organic frameworks,” Faraday Discuss., vol. 225, pp. 271–285, 2020.

The design of MicMec is intentionally similar to the design of [Yaff](https://github.com/molmod/yaff), a simulation package for atomistic systems.
In the process of creating MicMec, the original micromechanical model was modified slightly, to ensure user friendliness, accuracy and flexibility.
An overview of the model and key results were published in:

> J. Vandewalle, J. S. De Vos and S. M. J. Rogge, "MicMec: Developing the Micromechanical Model to Investigate the Mechanics of Correlated Node Defects in UiO-66," J. Phys. Chem. C, 2023.

The `simulations` directory contains command-line scripts for different types of simulations, including geometry optimizations and molecular dynamics simulations.
The rest of the MicMec API is explained in detail [here](https://molmod.github.io/micmec/).


## Installation

Clone this repository to an empty target directory of your liking and use `pip` to install the MicMec package.

```bash
git clone https://github.com/Jlvdwall/micmec.git <target>
cd <target>
pip install .
```

Currently, there are no unit tests for MicMec.
However, it is possible to verify the installation by running a simulation.

```bash
cd simulations
python md.py ../data/3x3x3_test_micmec.chk test.h5
```

(Please note that running the simulation from the root directory of this repository is not possible, as Python will attempt to find the `micmec.pes.ext` compiled module in the local `micmec` directory.
That module cannot be found locally, as it is only present in the install directory.)
