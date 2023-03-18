![THE MICROMECHANICAL PROCEDURE](https://github.com/Jlvdwall/micmec/blob/main/docs/figs/micmec_proc_wide.png)

MicMec is a geometry optimization and molecular dynamics code for coarse-grained crystalline materials. 
It is the first implementation of the micromechanical model, ever. 
The theoretical groundwork of the micromechanical model was originally established in:

> S. M. J. Rogge, “The micromechanical model to computationally investigate cooperative and correlated phenomena in metal-organic frameworks,” Faraday Discuss., vol. 225, pp. 271–285, 2020.

The design of MicMec is intentionally similar to the design of [Yaff](https://github.com/molmod/yaff), a simulation package for atomistic systems. 
In the process of creating MicMec, the original micromechanical model was modified slightly, to ensure user friendliness, accuracy and flexibility. 

The `simulations` directory contains command-line scripts for different types of simulations, including geometry optimizations and molecular dynamics simulations.
The rest of the MicMec API is explained in detail [here](https://jlvdwall.github.io/micmec/).


## Installation

```bash
git clone https://github.com/Jlvdwall/micmec.git
cd micmec
pip install .
```

