"""Model registry with ``forge://`` URI support.

MLflow is lazy-imported — this module can be imported without MLflow
installed.  Names are re-exported via ``__getattr__``.
"""

__all__ = ["ModelRegistry", "parse_forge_uri"]


def __getattr__(name: str):  # noqa: ANN001
    if name in __all__:
        from modules.forge.models import mlflow as _mlflow

        return getattr(_mlflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
