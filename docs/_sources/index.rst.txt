MicMec 1.0
##########

The micromechanical model is a coarse-grained force field model to simulate the mechanical behaviour of crystalline materials on a large length scale. MicMec is the first implementation of that model. The theoretical groundwork was originally established in: ::

    S. M. J. Rogge, “The micromechanical model to computationally investigate cooperative and 
    correlated phenomena in metal-organic frameworks,” Faraday Discuss., vol. 225, pp. 271–285, 2020.

MicMec is, essentially, a simulation package for coarse-grained, micromechanical systems. Its design is intentionally similar to `Yaff <https://github.com/molmod/yaff>`_, a simulation package for atomistic systems. In the process of building MicMec, the original micromechanical model was modified slightly, to ensure user friendliness, accuracy and flexibility. Recent changes and quality-of-life improvements are listed in the user guide and reference guide.


User Guide
==========

This guide serves as an introduction to MicMec.

.. toctree::
   :maxdepth: 2
   :numbered:
 
   installation.rst
   creating_types.rst
   constructing_systems.rst
   performing_simulations.rst


Reference Guide
===============

This guide is generated automatically based on the docstrings in the source code.

.. toctree::
   :maxdepth: 2
   :numbered:
 
   rg_micmec.rst
   rg_micmec_builder.rst
   rg_micmec_pes.rst
   rg_micmec_sampling.rst
   rg_micmec_analysis.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
