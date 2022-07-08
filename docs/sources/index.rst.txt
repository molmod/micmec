MicMec 1.0
##########

The micromechanical model is a coarse-grained force field model to simulate the mechanical behaviour of crystalline materials on a large length scale. MicMec is the first implementation of the micromechanical model, ever. The theoretical groundwork of the model was originally established in: ::

    S. M. J. Rogge, “The micromechanical model to computationally investigate cooperative and 
    correlated phenomena in metal-organic frameworks,” Faraday Discuss., vol. 225, pp. 271–285, 2020.

The micromechanical model has been the main topic of my master's thesis at the `Center for Molecular Modeling <http://molmod.ugent.be/>`_ (CMM). MicMec is, essentially, a simulation package for micromechanical systems. Its architecture is intentionally similar to `Yaff <https://github.com/molmod/yaff>`_, a simulation package for atomistic systems, also developed at the CMM. In the process of building MicMec, the original micromechanical model was modified slightly, to ensure user friendliness, accuracy and flexibility. All major changes with respect to the original model are listed in the text of my `master's thesis <https://github.com/Jlvdwall/micmec/blob/main/docs/thesis/>`_. More recent changes and quality-of-life improvements are listed in the user guide and reference guide.


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
