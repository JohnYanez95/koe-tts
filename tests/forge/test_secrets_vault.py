"""Test suite for forge.secrets.vault — Vault KV v2 HTTP client.

All tests mock urllib.request — no real Vault connection needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from modules.forge.secrets.vault import VaultClient, VaultError

# ════════════════════════════════════════════════════════════════════════
# 1. Client construction
# ════════════════════════════════════════════════════════════════════════


class TestVaultClient:
    def test_defaults(self):
        client = VaultClient()
        assert client.addr == "http://localhost:8200"
        assert client.timeout == 30

    def test_explicit_params(self):
        client = VaultClient(
            addr="http://vault:8200",
            token="s.my-token",
            timeout=10,
        )
        assert client.addr == "http://vault:8200"
        assert client.token == "s.my-token"
        assert client.timeout == 10

    def test_strips_trailing_slash(self):
        client = VaultClient(addr="http://vault:8200/")
        assert client.addr == "http://vault:8200"

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("VAULT_ADDR", "http://test-vault:8200")
        monkeypatch.setenv("VAULT_TOKEN", "s.test-token")
        client = VaultClient.from_env()
        assert client.addr == "http://test-vault:8200"
        assert client.token == "s.test-token"

    def test_explicit_timeout_overrides_module_default(self):
        """Explicit timeout param takes precedence over VAULT_TIMEOUT."""
        client = VaultClient(timeout=5)
        assert client.timeout == 5

    def test_module_timeout_constant_read(self):
        """VAULT_TIMEOUT module-level constant is a float."""
        import modules.forge.secrets.vault as vault_mod

        assert isinstance(vault_mod.VAULT_TIMEOUT, float)


# ════════════════════════════════════════════════════════════════════════
# 2. KV v2 path conversion
# ════════════════════════════════════════════════════════════════════════


class TestKV2ApiPath:
    def test_standard_path(self):
        assert VaultClient._kv2_api_path("secret/forge/minio") == "secret/data/forge/minio"

    def test_deep_path(self):
        assert VaultClient._kv2_api_path("secret/a/b/c") == "secret/data/a/b/c"

    def test_custom_mount(self):
        assert VaultClient._kv2_api_path("kv/myapp/config") == "kv/data/myapp/config"

    def test_strips_leading_slash(self):
        assert VaultClient._kv2_api_path("/secret/forge/minio") == "secret/data/forge/minio"

    def test_single_component_rejected(self):
        with pytest.raises(VaultError, match="Invalid secret path"):
            VaultClient._kv2_api_path("secret")


# ════════════════════════════════════════════════════════════════════════
# 3. get_secret
# ════════════════════════════════════════════════════════════════════════


class TestGetSecret:
    def _mock_response(self, data: dict, status: int = 200) -> MagicMock:
        """Create a mock urllib response."""
        body = json.dumps({"data": {"data": data}}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.status = status
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("urllib.request.urlopen")
    def test_reads_secret(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({"user": "admin", "password": "s3cret"})

        client = VaultClient(token="s.tok")
        data = client.get_secret("secret/forge/minio")

        assert data == {"user": "admin", "password": "s3cret"}
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "/v1/secret/data/forge/minio" in req.full_url
        assert req.get_header("X-vault-token") == "s.tok"

    @patch("urllib.request.urlopen")
    def test_no_token_header_when_unset(self, mock_urlopen):
        mock_urlopen.return_value = self._mock_response({"key": "val"})

        client = VaultClient(token=None)
        client.get_secret("secret/forge/test")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-vault-token") is None


# ════════════════════════════════════════════════════════════════════════
# 4. get_field
# ════════════════════════════════════════════════════════════════════════


class TestGetField:
    @patch("urllib.request.urlopen")
    def test_extracts_field(self, mock_urlopen):
        body = json.dumps({"data": {"data": {"user": "admin", "pass": "s3cret"}}}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        client = VaultClient(token="s.tok")
        assert client.get_field("secret/forge/minio", "user") == "admin"

    @patch("urllib.request.urlopen")
    def test_missing_field_raises(self, mock_urlopen):
        body = json.dumps({"data": {"data": {"user": "admin"}}}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        client = VaultClient(token="s.tok")
        with pytest.raises(VaultError, match="Field 'password' not found"):
            client.get_field("secret/forge/minio", "password")


# ════════════════════════════════════════════════════════════════════════
# 5. is_available
# ════════════════════════════════════════════════════════════════════════


class TestIsAvailable:
    @patch("urllib.request.urlopen")
    def test_returns_true_on_200(self, mock_urlopen):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        client = VaultClient()
        assert client.is_available() is True

    @patch("urllib.request.urlopen", side_effect=Exception("connection refused"))
    def test_returns_false_on_error(self, mock_urlopen):
        client = VaultClient()
        assert client.is_available() is False


# ════════════════════════════════════════════════════════════════════════
# 6. Error handling
# ════════════════════════════════════════════════════════════════════════


class TestVaultErrors:
    @patch("urllib.request.urlopen")
    def test_401_raises(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8200/v1/secret/data/test",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=None,
        )

        client = VaultClient(token="bad-token")
        with pytest.raises(VaultError, match="authentication failed"):
            client.get_secret("secret/test")

    @patch("urllib.request.urlopen")
    def test_403_raises(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8200/v1/secret/data/forbidden",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=None,
        )

        client = VaultClient(token="s.tok")
        with pytest.raises(VaultError, match="access denied"):
            client.get_secret("secret/forbidden")

    @patch("urllib.request.urlopen")
    def test_404_raises(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="http://localhost:8200/v1/secret/data/missing",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

        client = VaultClient(token="s.tok")
        with pytest.raises(VaultError, match="not found"):
            client.get_secret("secret/missing")

    @patch("urllib.request.urlopen")
    def test_connection_error_raises(self, mock_urlopen):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = VaultClient()
        with pytest.raises(VaultError, match="Cannot connect"):
            client.get_secret("secret/test")

    @patch("urllib.request.urlopen")
    def test_bad_response_structure(self, mock_urlopen):
        body = json.dumps({"unexpected": "format"}).encode()
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        client = VaultClient(token="s.tok")
        with pytest.raises(VaultError, match="Unexpected response"):
            client.get_secret("secret/test")
