# LD-GCN
LD-GCN is a library that implements a graph-based, LDNet-like architecture for nonlinear model order reduction of time-dependent parameterized PDEs.
This library was developed starting from [GCA-ROM](https://github.com/fpichi/gca-rom/tree/main).

## Installation
Before installing the package, you might want to create a dedicated `conda` environment.
The following line creates an environment and installs all the packages that are required:
```bash
conda env create -f environment.yml
```

## Tutorials
The `notebooks` folder contains four tutorials to show some of the functionalities of the library.

## Dataset generation
The datasets are not included in this repo since they are too heavy.
To create them, it is necessary to install [FEniCS](https://fenicsproject.org/download/archive/) and [RBniCS](https://www.rbnicsproject.org/).

## Cite LD-GCN
If you use LD-GCN for you research, you are encouraged to cite the paper using
```
@misc{tomada2026latentdynamicsgraphconvolutional,
      title={Latent Dynamics Graph Convolutional Networks for model order reduction of parameterized time-dependent PDEs}, 
      author={Lorenzo Tomada and Federico Pichi and Gianluigi Rozza},
      year={2026},
      eprint={2601.11259},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.11259}, 
}
```

## Authors
- Lorenzo Tomada (ltomada@sissa.it)
- Federico Pichi (fpichi@sissa.it)

in collaboration with Prof. Gianluigi Rozza (grozza@sissa.it) at SISSA mathLab.

The authors would like to acknowledge the valuable support provided by Francesco Sala and Mariella Kast.
