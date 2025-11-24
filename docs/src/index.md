**rEAMember** is an implementation of the computational model described in the paper ["Imagery in the entropic associative memory"](https://www.nature.com/articles/s41598-023-36761-6) by Luis A. Pineda et al.

> The Entropic Associative Memory (EAM) is a novel computational memory model in which functions representing arbitrary concrete or abstract objects are stored in a bi-dimensional array or table, called Associative Memory Register (AMR), which is used as the representational medium. The columns and the rows stand for the arguments and their values, respectively, and the functional relation is represented by filling up the cell at the corresponding intersection, for all the columns. Hence, every object is stored by marking up one cell of each column in the AMR, and can be thought of as a memory trace.

## Features

- Implementation of EAM and related neural architectures.
- Configuration-driven experiments with YAML files.
- Integration with PyTorch, TensorBoard, and other ML tools.
- Training and evaluation scripts for autoencoders and classifiers.
- Support for multiple datasets
- Embedding generation and visualization.

## Installation

rEAMember is currently in beta and not distributed as a package. To use it:

Clone the repository:
```bash
git clone https://github.com/yourusername/rEAMember.git
cd rEAMember
```

## Usage

Run commands using the management script:

```bash
uv run manage.py [command]
```

For example, to generate embeddings:
```bash
uv run manage.py get-embeddings --config ./config/spots-256.yml
```

To see all available commands:
```bash
uv run manage.py --help
```

## Repository Structure

- `reamember/` — Core library and modules
  - `eam/` — Associative memory implementation
  - `neuralnets/` — Neural network models (autoencoder, classifier, transformer)
- `config/` — Experiment and dataset configuration files
- `data/` — Datasets and raw data
- `experiments/` — Experiment outputs and logs
- `logs/` — TensorBoard logs and embeddings
- `docs/` — Documentation (MkDocs)
- `manage.py` — Main command-line interface