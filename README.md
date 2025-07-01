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

Run: 

```bash
uv run manage.py --help
```

## Docs

```bash
uv run pdoc manage.py reamember
```