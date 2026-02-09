"""Pure HTTP client for HashiCorp Vault KV v2 secrets.

Uses stdlib ``urllib.request`` — no ``requests`` dependency.  Designed to
replace the subprocess-based ``koe bootstrap`` with safe, pure-Python
secret retrieval.

Timeout Limitation
------------------
``urllib.request.urlopen`` accepts a single ``timeout`` float covering
both connect and read.  For localhost Vault this is fine — connect is
effectively instant.  The timeout defaults to 30 s via
``FORGE_VAULT_TIMEOUT``.

KV v2 Path Convention
---------------------
Callers use CLI-style paths (e.g. ``secret/forge/minio``).  The first
component is the mount, and ``/data/`` is injected automatically::

    secret/forge/minio  →  GET {addr}/v1/secret/data/forge/minio

Environment Variables
---------------------
- ``VAULT_ADDR``            — Vault address (default: http://localhost:8200)
- ``VAULT_TOKEN``           — Vault token for authentication
- ``FORGE_VAULT_TIMEOUT``   — Request timeout in seconds (default: 30)

Public API
----------
- ``VaultError`` — base exception for Vault operations
- ``VaultClient`` — HTTP client with get_secret / get_field / is_available
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

VAULT_TIMEOUT: float = float(os.getenv("FORGE_VAULT_TIMEOUT", "30"))


class VaultError(Exception):
    """Base error for Vault operations."""


class VaultClient:
    """Pure-HTTP Vault KV v2 client.

    Parameters
    ----------
    addr
        Vault server address.  Falls back to ``VAULT_ADDR`` env var,
        then ``http://localhost:8200``.
    token
        Vault token.  Falls back to ``VAULT_TOKEN`` env var.
    timeout
        Request timeout in seconds (single float for urllib).
        Falls back to ``FORGE_VAULT_TIMEOUT`` env var, then 30.
    """

    def __init__(
        self,
        addr: str | None = None,
        token: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.addr = (addr or os.getenv("VAULT_ADDR", "http://localhost:8200")).rstrip("/")
        self.token = token or os.getenv("VAULT_TOKEN")
        self.timeout = timeout if timeout is not None else VAULT_TIMEOUT

    @classmethod
    def from_env(cls) -> VaultClient:
        """Construct a VaultClient from environment variables."""
        return cls()

    @staticmethod
    def _kv2_api_path(cli_path: str) -> str:
        """Convert CLI-style KV path to API path with /data/ injected.

        ``secret/forge/minio`` → ``secret/data/forge/minio``
        """
        parts = cli_path.strip("/").split("/", 1)
        if len(parts) < 2:
            raise VaultError(
                f"Invalid secret path: {cli_path!r}. "
                "Expected format: mount/path (e.g. secret/forge/minio)"
            )
        mount, remainder = parts
        return f"{mount}/data/{remainder}"

    def _request(self, path: str) -> dict:
        """Make an authenticated GET request to Vault.

        Returns parsed JSON response body.
        """
        url = f"{self.addr}/v1/{path}"

        headers = {}
        if self.token:
            headers["X-Vault-Token"] = self.token

        req = urllib.request.Request(url, headers=headers, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read()
                return json.loads(body)
        except urllib.error.HTTPError as e:
            status = e.code
            if status == 401:
                raise VaultError("Vault authentication failed (401)") from e
            if status == 403:
                raise VaultError(f"Vault access denied (403) for path: {path}") from e
            if status == 404:
                raise VaultError(f"Secret not found (404): {path}") from e
            raise VaultError(f"Vault HTTP error {status} for path: {path}") from e
        except urllib.error.URLError as e:
            raise VaultError(f"Cannot connect to Vault at {self.addr}: {e.reason}") from e

    def get_secret(self, path: str) -> dict:
        """Read a KV v2 secret.

        Parameters
        ----------
        path
            CLI-style secret path (e.g. ``secret/forge/minio``).
            The first component is the mount; ``/data/`` is injected
            automatically for the KV v2 API.

        Returns
        -------
        dict
            The secret data (inner ``data`` dict from KV v2 response).

        Raises
        ------
        VaultError
            If the secret cannot be read (auth, permission, not found,
            connection error).
        """
        api_path = self._kv2_api_path(path)
        response = self._request(api_path)

        try:
            return response["data"]["data"]
        except (KeyError, TypeError) as e:
            raise VaultError(
                f"Unexpected response structure for {path}"
            ) from e

    def get_field(self, path: str, field: str) -> str:
        """Read a single field from a KV v2 secret.

        Parameters
        ----------
        path
            CLI-style secret path (e.g. ``secret/forge/minio``).
        field
            Field name within the secret (e.g. ``user``).

        Returns
        -------
        str
            The field value.

        Raises
        ------
        VaultError
            If the secret or field cannot be read.
        """
        data = self.get_secret(path)
        if field not in data:
            raise VaultError(
                f"Field {field!r} not found in secret {path!r}. "
                f"Available fields: {sorted(data.keys())}"
            )
        return str(data[field])

    def is_available(self) -> bool:
        """Check if Vault is reachable via health endpoint.

        Returns True if Vault responds to ``/v1/sys/health``, False
        otherwise.  Does not require authentication.
        """
        url = f"{self.addr}/v1/sys/health"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except Exception:
            return False
