# rEAMember

This project is an implementation of the paper "Imagery in the entropic associative memory" by [Luis A. Pineda et al.](https://www.nature.com/articles/s41598-023-36761-6).

> The Entropic Associative Memory (EAM) is a novel computational memory model in which functions representing arbitrary concrete or abstract objects are stored in a bi-dimensional array or table, called Associative Memory Register (AMR), which is used as the representational medium. The columns and the rows stand for the arguments and their values, respectively, and the functional relation is represented by filling up the cell at the corresponding intersection, for all the columns. Hence, every object is stored by marking up one cell of each column in the AMR, and can be thought of as a memory trace. 

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