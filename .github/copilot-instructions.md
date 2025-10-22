# Copilot Instructions

Welcome, Copilot!  
This repository is for a Python toolset that generates and compare array and tabular data reading performances.

---

## General Guidelines

- **Language & Style:**  
  - Use idiomatic Python (3.8+).  
  - Prefer clear NumPy or Google-style docstrings.  
  - Write modular code with clear separation of concerns.
  - Follow style guidelines as in PEP 8, PEP 257. Each python file should have a grade of 8 or more when run through `pylint`.
  - Follow PEP 484 for type hinting.
  - The code should compatible with any version of Python 3.12 or newer.

- **Structure & Naming:**  
  - All code to generate the test datasets is under `data_sparsity/`
  - Use CamelCase for classes, snake_case for functions/variables.
  - Keep the public API minimal and clean.
  - Keep in mind that this code is used by non-experienced Python users too, so any design solution that is advanced should be justified and well documented and/or commented.
  - Modules containing classes should have their name match the class name, e.g. the class `MyClass` would be defined in `my_class.py`, the subclass `MyClassSubclass` would be in `my_class_subclass.py`
  - Whenever possible, a subclass name should add from the parent class, making inheritance clear. For example, `MyClassSubclass` would inherit from `MyClass`. Alternatives like `MySubclass` or `MySubclassClass` are discouraged.

- **Dependencies:**
  - Generate and keep up to date both a conda `environment.yml` file and a python `pyproject.toml`.
  - Propose/add new dependencies only when necessary and document why in PRs.

---

## Command-Line Interface

- CLI entry points should live in `FormatAnalyzer/cli/`.

## Documentation

- Keep `README.md` updated with new features and usage.
- All user-facing functions/scripts must have clear docstrings.
- If adding new configuration options, update any relevant YAML examples.

---

## Testing & Validation

- Prefer pure Python and in-memory tests; avoid committing large data files.
- New features should include at least one minimal test or usage example (in code, docstring, or README).

---

## File and Directory Conventions

- Do **not** commit generated data files (e.g., `.parquet`, `.pkl`, `.nc`, `.csv`, etc.) or large outputs.
- Respect and update `.gitignore` as needed.

---

## Communication

- Prioritize clarity and maintainability over cleverness.
- If a design decision is not obvious, leave a code comment explaining your reasoning.

---

Thank you for helping make this repo robust and future-proof!
