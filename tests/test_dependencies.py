import importlib

import pytest


DEPENDENCIES = [
    ("pandas", "pandas", "Data loading and event utilities"),
    ("numpy", "numpy", "Numeric helpers and modeling"),
    ("sklearn", "scikit-learn", "All classifiers/regressors and preprocessing"),
    ("seaborn", "seaborn", "Plot styling in plot_generation.py"),
    ("matplotlib", "matplotlib", "Plot saving and diagnostics"),
    ("lightgbm", "lightgbm", "Required for modeling_pipeline.py"),
]


@pytest.mark.parametrize("module_name,pip_name,reason", DEPENDENCIES)
def test_core_dependencies_installed(module_name: str, pip_name: str, reason: str) -> None:
    """
    Fail early when a required dependency is missing so the notebooks/scripts don't break later.
    """
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - triggers only when missing/broken
        extra_hint = ""
        if module_name == "lightgbm":
            extra_hint = (
                " On macOS you may also need `brew install libomp` before reinstalling lightgbm."
            )
        pytest.fail(
            f"Missing or broken dependency '{module_name}' ({reason}). "
            f"Install with `python3 -m pip install {pip_name}` (see README).{extra_hint} "
            f"Import error: {exc}"
        )
