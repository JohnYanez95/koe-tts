"""MLflow model registry wrapper with ``forge://`` URI support.

Provides a high-level ``ModelRegistry`` class that translates custom
``forge://`` URIs into standard MLflow model URIs.  MLflow is
lazy-imported — this module can be imported without MLflow installed.

URI Scheme
----------
``forge://name@ref`` is translated to MLflow format:

- ``forge://vits@prod``  → ``models:/vits@prod``   (alias)
- ``forge://vits@best``  → ``models:/vits@best``   (alias)
- ``forge://vits@3``     → ``models:/vits/3``      (version number)
- ``forge://vits@v3``    → ``models:/vits/3``      (version number, strip 'v')
- ``models:/...``        → passthrough (already MLflow format)

Environment Variables
---------------------
- ``MLFLOW_TRACKING_URI`` — MLflow tracking server (default: http://localhost:5002)

Public API
----------
- ``ModelRegistry`` — high-level registry client
- ``parse_forge_uri(uri)`` — URI translation (module-level function)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FORGE_URI_RE = re.compile(r"^forge://([^@]+)@(.+)$")
_VERSION_RE = re.compile(r"^v?(\d+)$")


def parse_forge_uri(uri: str) -> str:
    """Translate a ``forge://`` URI to an MLflow model URI.

    Parameters
    ----------
    uri
        Either a ``forge://name@ref`` URI or an MLflow ``models:/...``
        URI (passed through unchanged).

    Returns
    -------
    str
        MLflow-compatible model URI.

    Raises
    ------
    ValueError
        If the URI format is invalid.

    Examples
    --------
    >>> parse_forge_uri("forge://vits@prod")
    'models:/vits@prod'
    >>> parse_forge_uri("forge://vits@3")
    'models:/vits/3'
    >>> parse_forge_uri("forge://vits@v3")
    'models:/vits/3'
    >>> parse_forge_uri("models:/vits@best")
    'models:/vits@best'
    """
    if uri.startswith("models:/"):
        return uri

    match = _FORGE_URI_RE.match(uri)
    if not match:
        raise ValueError(
            f"Invalid model URI: {uri!r}. "
            "Expected forge://name@ref or models:/name/version"
        )

    name, ref = match.group(1), match.group(2)

    # Check if ref is a version number (e.g. "3" or "v3")
    version_match = _VERSION_RE.match(ref)
    if version_match:
        return f"models:/{name}/{version_match.group(1)}"

    # Otherwise treat as alias
    return f"models:/{name}@{ref}"


class ModelRegistry:
    """MLflow model registry with ``forge://`` URI support.

    Parameters
    ----------
    tracking_uri
        MLflow tracking server URI.  Falls back to
        ``MLFLOW_TRACKING_URI`` env var, then ``http://localhost:5002``.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5002"
        )

    @classmethod
    def from_env(cls) -> ModelRegistry:
        """Construct a ModelRegistry from environment variables."""
        return cls()

    def _configure(self) -> None:
        """Set MLflow tracking URI.  Lazy-imports mlflow."""
        try:
            import mlflow
        except ImportError:
            raise ImportError(
                "mlflow is required for model registry operations. "
                "Install with: pip install mlflow"
            ) from None

        mlflow.set_tracking_uri(self.tracking_uri)

    def load_model(self, uri: str) -> Any:
        """Load a model by URI.

        Supports ``forge://`` and ``models:/`` URIs.

        Parameters
        ----------
        uri
            Model URI (``forge://name@ref`` or ``models:/name/version``).

        Returns
        -------
        Any
            Loaded model (MLflow PyFunc wrapper).
        """
        import mlflow.pyfunc

        self._configure()
        mlflow_uri = parse_forge_uri(uri)
        logger.info("Loading model: %s → %s", uri, mlflow_uri)
        return mlflow.pyfunc.load_model(mlflow_uri)

    def register(
        self,
        model_path: Path | str,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> Any:
        """Register a model artifact in the registry.

        Parameters
        ----------
        model_path
            Path to model artifact (checkpoint file).
        name
            Registered model name.
        tags
            Optional version tags.

        Returns
        -------
        Any
            MLflow ModelVersion object.
        """
        import mlflow

        self._configure()
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return mlflow.register_model(
            model_uri=str(model_path),
            name=name,
            tags=tags,
        )

    def list_models(self) -> list[dict]:
        """List all registered models.

        Returns
        -------
        list[dict]
            List of dicts with ``name``, ``description``, and
            ``latest_versions`` keys.
        """
        import mlflow

        self._configure()
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()

        return [
            {
                "name": m.name,
                "description": m.description or "",
                "latest_versions": [
                    {"version": v.version, "aliases": v.aliases}
                    for v in (m.latest_versions or [])
                ],
            }
            for m in models
        ]
