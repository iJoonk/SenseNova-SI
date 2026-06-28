# Repository Guidelines

## Project Structure & Module Organization

`sensenova_si/` contains the inference package and model adapters for InternVL, Qwen, and BAGEL. `example.py` and `example_bagel.py` are runnable entry points for smoke testing released models. `examples/` stores sample images and generated outputs used by the docs. `config/` holds default runtime configuration. `docs/en/` and `docs/zh/` contain usage examples. Training code is split by backend under `training/intern_vl/`, `training/qwen3_vl/`, and `training/bagel/`; `training/lmms-engine/` is a git submodule used by Qwen3-VL training.

## Build, Test, and Development Commands

- `uv sync --extra cu124`: create the inference environment; replace `cu124` with the CUDA extra listed in `pyproject.toml`.
- `source .venv/bin/activate`: activate the local environment.
- `python example.py --question "Hello" --model_path sensenova/SenseNova-SI-1.4-InternVL3-8B`: run a minimal inference smoke test.
- `python example_bagel.py --model_path sensenova/SenseNova-SI-1.1-BAGEL-7B-MoT --prompt "..." --mode generate`: test BAGEL generation.
- `ruff check . --select I` and `ruff format . --check`: match CI import-order and formatting checks.
- `git submodule update --init --recursive`: initialize `training/lmms-engine/` before Qwen3-VL training work.

## Coding Style & Naming Conventions

Use Python 3.11 for the package unless a training subtree documents Python 3.10. Follow Ruff formatting, 4-space indentation, sorted imports, and existing module patterns. Name Python modules and functions in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Keep model-specific logic in the matching adapter or training subtree rather than adding cross-backend conditionals to unrelated files.

## Testing Guidelines

There is no dedicated test suite in this repository. For code changes, run Ruff and at least one targeted smoke test that exercises the touched path, such as `example.py` for inference adapters or the relevant preprocessing script in `training/qwen3_vl/`. Document any skipped GPU/model-download validation in the pull request.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, sometimes with Conventional Commit prefixes, for example `fix: add preprocessing script...` or `Update README`. Keep commits focused and mention the affected component when useful. Pull requests should include a concise summary, validation commands, linked issue or model/data release context when applicable, and screenshots or sample outputs for user-visible generation or documentation changes.

## Security & Configuration Tips

Do not commit downloaded datasets, pretrained checkpoints, generated caches, API tokens, or local environment folders. Place large external assets under ignored runtime paths such as `training/data/` or `training/pretrained_models/`, and reference download commands instead of storing binaries in git.
