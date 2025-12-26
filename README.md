# Agenthle

Agenthle is ...

## Structure

```
agenthle/
├── submodules/
│   └── OSWorld/          # OSWorld submodule
├── gcp_vm_management/    # GCP VM management tools
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone git@github.com:cua-verse/agenthle.git
cd agenthle
```

Or clone with submodules:

```bash
git clone --recursive git@github.com:cua-verse/agenthle.git
cd agenthle
```

### 2. Initialize and update submodules

If you cloned without submodules:

```bash
git submodule update --init --recursive
```

### 3. Install uv

If you don't have uv installed:

```bash
# on mac
brew install uv
```

Or follow the [official installation guide](https://github.com/astral-sh/uv#installation).

### 4. Install dependencies using uv workspace

This project uses [uv workspace](https://docs.astral.sh/uv/workspaces/) to manage dependencies. The workspace includes both `agenthle` and the `OSWorld` submodule.

Install all dependencies:

```bash
uv sync
```

This will:
- Create a virtual environment in `.venv`
- Install all dependencies from both `agenthle` and `OSWorld` packages
- Install all packages in editable mode

## Usage

### Running scripts

Use `uv run` to execute Python scripts in the workspace:

```bash
uv run python gcp_vm_management/start_persistent_vm.py
```

Or activate the virtual environment manually:

```bash
source .venv/bin/activate
python gcp_vm_management/start_persistent_vm.py
```

### Working with submodules

**Check submodule status:**

```bash
git submodule status
```

**Update all submodules:**

```bash
git submodule update --remote
```

**Update OSWorld to the latest version:**

```bash
cd submodules/OSWorld
git pull origin main  # or master, depending on the branch
cd ../..
git add submodules/OSWorld
git commit -m "Update OSWorld submodule"
```

For detailed usage instructions, see `gcp_vm_management/PERSISTENT_VM_USAGE.md`.

## License

See LICENSE file for details.

