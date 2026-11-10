"""Run each package smoke module in sorted filename order."""

import importlib
from types import ModuleType

from public_label_rework.common import ROOT
from retraining_data.schema import DataError


def smoke_main(module: ModuleType) -> None:
    """Invoke a dynamically discovered smoke module."""
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise DataError(f"{module.__name__}: missing callable main")
    _ = entry()


def main() -> None:
    """Discover and run all smoke modules, preserving every failure."""
    package = ROOT / "public_label_rework"
    modules = sorted(path.stem for path in package.glob("smoke_*.py"))
    failures: list[str] = []
    for name in modules:
        module = importlib.import_module(f"public_label_rework.{name}")
        try:
            smoke_main(module)
        except (DataError, OSError, ValueError) as error:
            failures.append(f"{name}: {error}")
            continue
        print(f"PASS {name}")
    if failures:
        raise DataError("; ".join(failures))


if __name__ == "__main__":
    main()
