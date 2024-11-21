# wd-ai-hackathon

## Installation

When running as devcontainer or codespace everything should be set up already.

When running locally `uv`needs to be installed:

```
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/uv/releases/download/0.5.4/uv-installer.sh | sh

```

after installation run 

```
uv sync
```

Also following vscode extensions are needed:

```
"ms-python.python"
"ms-toolsai.jupyter"
"ms-toolsai.jupyter-renderers"
"ms-toolsai.jupyter-keymap"
```

## Sanity check

Open `notebooks/00_hello_world.ipynb` and select `.venv`as kernel/virtual environment.

Execute the notebook to make sure it runs without error.

## Hackathon

### Introduction to Semantic Kernel

### Infuse knowledge with the RAG pattern

### Orchestrate knowledge with agents

### ???