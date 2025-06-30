# rEAMember

This project is an implementation of the paper "Imagery in the entropic associative memory" by [Luis A. Pineda et al.](https://www.nature.com/articles/s41598-023-36761-6).

## Usage

```bash
uv run manage.py [command]
```

For example:

```bash
uv run manage.py get-embeddings --config ./config/spots-256.yml
```
This command will generate embeddings for the dataset specified in the configuration file `spots-256.yml`.

## Available Commands

Main

- `get-embeddings`: Generate embeddings for the dataset specified in the configuration file.
- `train-autoencoder`: Train the autoencoder model with the specified configuration file.
- `train-classifier`: Train the classifier model with the specified configuration file.

Utils

- `run-tensorboard`: Start TensorBoard to visualize training metrics.
- `clean-logs`: Clean the logs generated during experiments.

## Docs

```bash
uv run pdoc manage.py reamember
```