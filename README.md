# wd-ai-hackathon

## Installation

When running as devcontainer or codespace everything should be set up already.

When running locally `uv`needs to be installed:

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/uv/releases/download/0.5.4/uv-installer.sh | sh

```

after installation run

```sh
uv sync
```

Also following vscode extensions are needed:

```sh
"ms-python.python"
"ms-toolsai.jupyter"
"ms-toolsai.jupyter-renderers"
"ms-toolsai.jupyter-keymap"
"charliermarsh.ruff"
"GitHub.copilo"
"GitHub.copilo-chat"
```

## Sanity check

Open `notebooks/00_hello_world.ipynb` and select `.venv`as kernel/virtual environment.

Execute the notebook to make sure it runs without error.

## AI Hackathon: Building Intelligent Applications with Semantic Kernel

This repository contains a series of Jupyter notebooks that guide you through building AI-powered applications using Semantic Kernel. The content progresses from basic concepts to advanced implementations, with each module containing both theoretical foundations and practical exercises.

### Prerequisites

- Python 3.8 or later
- Basic understanding of Python programming
- OpenAI API key or Azure OpenAI access
- Semantic Kernel library installed

### Notebooks Overview

#### Module 1: Introduction to Semantic Kernel

- **01_a_introduction_to_semantic_kernel.ipynb**: Theory module covering:
  - What is Semantic Kernel
  - Setting up the Kernel
  - Plugins (Native and Semantic)
  - Memory capabilities
  - Practical examples including an Emoji Translator
- **01_b_Introduction_to_semantic_kernel.ipynb**: Practical exercises for implementing small applications using core concepts:
  - Fusion-Chef: Recipe combination system
  - QnA Bot: Memory-based question answering
  - Roleplay Gamemaster: Interactive game system
  - Shell Meister: Natural language to shell command translator

#### Module 2: Planning and Execution

- **02_a_a_bad_plan_is_better_than_no_plan.ipynb**: Theory module exploring:
  - AI Planning concepts
  - Types of Planners (Sequential and Function Calling Stepwise)
  - Memory integration with planners
  - Web research capabilities
- **02_b_a_bad_plan_is_better_than_no_plan.ipynb**: Practical exercises including:
  - Code Documentation Assistant
  - Personal Finance Advisor
  - AI Dungeon Master
  - Research Paper Assistant

#### Module 3: AI Agents

- **03_a_agents.ipynb**: Theory module covering:
  - Introduction to SK Agents
  - Creating basic agents
  - Agent capabilities and plugins
  - Multi-agent interactions
- **03_b_agents.ipynb**: Practical exercises for building:
  - Code Review Team
  - Game Master System
  - Competitive Debate System
  - Technical Support System
  - Research Paper Collaboration System
- **03_c_battle_of_the_agents.ipynb**: Advanced concepts on:
  - Agent simulation and testing
  - Conversation modeling
  - Performance evaluation

#### Module 4: Process Management

- **04_a_its_a_process.ipynb**: Theory module on:
  - Building an AI Travel Agent
  - Process Steps and Event Handling
  - State Management
  - AI Integration
- **04_b_its_a_process.ipynb**: Practical exercises for implementing:
  - Smart Home Automation System
  - Restaurant Kitchen Process System
  - Document Approval Workflow
  - Data ETL Pipeline System
  - Project Management System

#### Module 5: Capstone Project

- **05_a_putting_it_all_together.ipynb**: Understanding Knowledge Graphs
  - Theory and foundations
  - Graph components and relationships
  - Visualization techniques
  - Best practices
- **05_b_putting_it_all_together.ipynb**: Building an AI Knowledge System
  - Core Knowledge Graph System
  - Knowledge Extraction Agents
  - Query Agent System
  - Process Integration
- **05_c_putting_it_all_together.ipynb**: Extended Practical Applications
  - Research Paper Assistant
  - Technical Documentation Assistant
  - Learning Path Generator
  - Troubleshooting Assistant
  - Content Recommendation System

### Learning Path

The notebooks are designed to be completed in order, as each module builds upon concepts introduced in previous modules. The progression goes from basic Semantic Kernel concepts to advanced multi-agent systems and process management, culminating in a comprehensive capstone project.

Each module contains:

- Theoretical foundations
- Code examples
- Practical exercises
- Best practices and tips
- Real-world applications
