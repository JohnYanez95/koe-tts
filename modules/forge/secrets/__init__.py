"""Secret management — Vault KV v2 client.

Public API
----------
- ``VaultClient`` — pure-HTTP Vault client (stdlib only)
- ``VaultError`` — base exception for Vault operations
"""

from modules.forge.secrets.vault import VaultClient, VaultError

__all__ = ["VaultClient", "VaultError"]
