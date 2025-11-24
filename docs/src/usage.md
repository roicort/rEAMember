# Usage

Currently, rEAMember is in beta release and it's not a package. You'll need to clone the repository.

## Manage.py

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
