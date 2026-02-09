"""Test suite for forge.storage.s3 — S3/MinIO storage backend.

All tests mock boto3 so no real S3 connection is needed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modules.forge.storage.protocols import NotFoundError, StorageBackend
from modules.forge.storage.s3 import (
    S3ConfigError,
    S3StorageBackend,
    S3UploadError,
    build_raw_zone_prefix,
    collect_upload_files,
    ensure_bucket,
    get_s3_client,
    get_s3_config,
    is_s3_available,
    upload_file,
    validate_path_component,
)

# ════════════════════════════════════════════════════════════════════════
# 1. validate_path_component
# ════════════════════════════════════════════════════════════════════════


class TestValidatePathComponent:
    def test_valid_simple(self):
        assert validate_path_component("my-bucket") == "my-bucket"

    def test_valid_with_dots(self):
        assert validate_path_component("v1.0.0") == "v1.0.0"

    def test_valid_with_equals(self):
        assert validate_path_component("corpus=jsut") == "corpus=jsut"

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            validate_path_component("")

    def test_traversal_rejected(self):
        with pytest.raises(ValueError, match="'\\.\\.'"):
            validate_path_component("foo/../bar")

    def test_absolute_path_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            validate_path_component("/etc/passwd")

    def test_backslash_rejected(self):
        with pytest.raises(ValueError, match="absolute"):
            validate_path_component("\\windows\\path")

    def test_control_chars_rejected(self):
        with pytest.raises(ValueError, match="control"):
            validate_path_component("foo\x00bar")

    def test_space_rejected(self):
        with pytest.raises(ValueError, match="allowed"):
            validate_path_component("foo bar")

    def test_unicode_rejected(self):
        with pytest.raises(ValueError, match="allowed"):
            validate_path_component("日本語")


# ════════════════════════════════════════════════════════════════════════
# 2. get_s3_config
# ════════════════════════════════════════════════════════════════════════


class TestGetS3Config:
    def test_configured(self, monkeypatch):
        monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
        monkeypatch.setenv("MINIO_ROOT_USER", "minioadmin")
        monkeypatch.setenv("MINIO_ROOT_PASSWORD", "minioadmin")

        config = get_s3_config()
        assert config is not None
        assert config["endpoint"] == "http://localhost:9000"
        assert config["access_key"] == "minioadmin"
        assert config["secret_key"] == "minioadmin"

    def test_missing_vars(self, monkeypatch):
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
        monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)

        assert get_s3_config() is None

    def test_partial_vars(self, monkeypatch):
        monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
        monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
        monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)

        assert get_s3_config() is None


# ════════════════════════════════════════════════════════════════════════
# 3. get_s3_client
# ════════════════════════════════════════════════════════════════════════


class TestGetS3Client:
    def test_unconfigured_returns_none(self, monkeypatch):
        monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
        monkeypatch.delenv("MINIO_ROOT_USER", raising=False)
        monkeypatch.delenv("MINIO_ROOT_PASSWORD", raising=False)

        assert get_s3_client() is None

    @patch("modules.forge.storage.s3.boto3", create=True)
    def test_configured_returns_client(self, mock_boto3, monkeypatch):
        monkeypatch.setenv("MINIO_ENDPOINT", "http://localhost:9000")
        monkeypatch.setenv("MINIO_ROOT_USER", "minioadmin")
        monkeypatch.setenv("MINIO_ROOT_PASSWORD", "minioadmin")

        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        # Patch import to succeed
        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()}):
            client = get_s3_client()

        assert client is not None


class TestGetS3ClientTimeouts:
    def test_timeout_defaults(self):
        from modules.forge.storage.s3 import (
            S3_CONNECT_TIMEOUT,
            S3_MAX_ATTEMPTS,
            S3_READ_TIMEOUT,
        )

        assert S3_CONNECT_TIMEOUT == int(os.getenv("FORGE_S3_CONNECT_TIMEOUT", "5"))
        assert S3_READ_TIMEOUT == int(os.getenv("FORGE_S3_READ_TIMEOUT", "30"))
        assert S3_MAX_ATTEMPTS == int(os.getenv("FORGE_S3_MAX_ATTEMPTS", "3"))

    def test_region_default(self):
        from modules.forge.storage.s3 import S3_REGION

        assert S3_REGION == os.getenv("FORGE_S3_REGION", "us-east-1")


# ════════════════════════════════════════════════════════════════════════
# 4. is_s3_available
# ════════════════════════════════════════════════════════════════════════


class TestIsS3Available:
    @patch("modules.forge.storage.s3.get_s3_client")
    def test_not_configured(self, mock_get_client):
        mock_get_client.return_value = None
        assert is_s3_available() is False

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_configured_reachable(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        assert is_s3_available() is True
        mock_client.list_buckets.assert_called_once()

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_configured_unreachable(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_buckets.side_effect = ConnectionError("refused")
        mock_get_client.return_value = mock_client
        assert is_s3_available() is False


# ════════════════════════════════════════════════════════════════════════
# 5. ensure_bucket
# ════════════════════════════════════════════════════════════════════════


class TestEnsureBucket:
    @patch("modules.forge.storage.s3.get_s3_client")
    def test_no_client(self, mock_get_client):
        mock_get_client.return_value = None
        assert ensure_bucket("forge") is False

    def test_bucket_exists(self):
        mock_client = MagicMock()
        assert ensure_bucket("forge", mock_client) is True
        mock_client.head_bucket.assert_called_once_with(Bucket="forge")

    def test_invalid_bucket_name(self):
        mock_client = MagicMock()
        with pytest.raises(ValueError):
            ensure_bucket("../../bad", mock_client)


# ════════════════════════════════════════════════════════════════════════
# 6. upload_file
# ════════════════════════════════════════════════════════════════════════


class TestUploadFile:
    def test_success(self, tmp_path):
        src = tmp_path / "test.wav"
        src.write_bytes(b"fake audio content")

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ETag": '"abc123"'}

        result = upload_file(
            local_path=src,
            bucket="forge",
            key="lake/raw/test.wav",
            client=mock_client,
        )

        assert result["key"] == "lake/raw/test.wav"
        assert result["bucket"] == "forge"
        assert result["s3_uri"] == "s3://forge/lake/raw/test.wav"
        assert result["size"] == 18
        assert result["etag"] == "abc123"
        assert result["checksum_sha256"] is not None
        mock_client.upload_file.assert_called_once()

    def test_missing_file(self, tmp_path):
        mock_client = MagicMock()
        with pytest.raises(FileNotFoundError):
            upload_file(
                local_path=tmp_path / "nonexistent.wav",
                bucket="forge",
                key="test.wav",
                client=mock_client,
            )

    def test_invalid_bucket(self, tmp_path):
        src = tmp_path / "test.wav"
        src.write_bytes(b"data")
        mock_client = MagicMock()

        with pytest.raises(ValueError):
            upload_file(
                local_path=src,
                bucket="../escape",
                key="test.wav",
                client=mock_client,
            )

    def test_upload_failure(self, tmp_path):
        src = tmp_path / "test.wav"
        src.write_bytes(b"data")

        mock_client = MagicMock()
        mock_client.upload_file.side_effect = Exception("network error")

        with pytest.raises(S3UploadError, match="Failed to upload"):
            upload_file(
                local_path=src,
                bucket="forge",
                key="test.wav",
                client=mock_client,
            )

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_no_client_raises(self, mock_get_client, tmp_path):
        mock_get_client.return_value = None
        src = tmp_path / "test.wav"
        src.write_bytes(b"data")

        with pytest.raises(S3ConfigError):
            upload_file(local_path=src, bucket="forge", key="test.wav")

    def test_skip_checksum(self, tmp_path):
        src = tmp_path / "test.wav"
        src.write_bytes(b"data")

        mock_client = MagicMock()
        mock_client.head_object.return_value = {"ETag": '"abc"'}

        result = upload_file(
            local_path=src,
            bucket="forge",
            key="test.wav",
            client=mock_client,
            compute_checksum=False,
        )
        assert result["checksum_sha256"] is None


# ════════════════════════════════════════════════════════════════════════
# 7. collect_upload_files
# ════════════════════════════════════════════════════════════════════════


class TestCollectUploadFiles:
    def test_basic(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"audio")
        (tmp_path / "b.txt").write_text("text")

        files = collect_upload_files(tmp_path)
        assert len(files) == 2

    def test_with_filter(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"audio")
        (tmp_path / "b.txt").write_text("text")

        files = collect_upload_files(tmp_path, file_filter="*.wav")
        assert len(files) == 1
        assert files[0].name == "a.wav"

    def test_max_files_exceeded(self, tmp_path):
        for i in range(5):
            (tmp_path / f"file{i}.txt").write_text(f"content{i}")

        with pytest.raises(ValueError, match="Too many files"):
            collect_upload_files(tmp_path, max_files=3)

    def test_symlink_escape_skipped(self, tmp_path):
        (tmp_path / "legit.txt").write_text("ok")

        # Create a symlink that escapes the directory
        escape_link = tmp_path / "escape.txt"
        try:
            escape_link.symlink_to("/etc/hostname")
        except OSError:
            pytest.skip("Cannot create symlinks on this platform")

        files = collect_upload_files(tmp_path)
        names = [f.name for f in files]
        assert "legit.txt" in names
        assert "escape.txt" not in names

    def test_not_a_directory(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("not a dir")

        with pytest.raises(NotADirectoryError):
            collect_upload_files(f)


# ════════════════════════════════════════════════════════════════════════
# 8. build_raw_zone_prefix
# ════════════════════════════════════════════════════════════════════════


class TestBuildRawZonePrefix:
    def test_valid_inputs(self):
        result = build_raw_zone_prefix(
            corpus="jsut",
            corpus_version="v1.0",
            domain="koe",
        )
        assert result == "lake/raw/koe/corpora/corpus=jsut/corpus_version=v1.0"

    def test_domain_required(self):
        # domain is a required positional arg — TypeError if missing
        with pytest.raises(TypeError):
            build_raw_zone_prefix(corpus="jsut", corpus_version="v1.0")  # type: ignore[call-arg]

    def test_invalid_corpus(self):
        with pytest.raises(ValueError):
            build_raw_zone_prefix(
                corpus="../bad",
                corpus_version="v1.0",
                domain="koe",
            )

    def test_invalid_version(self):
        with pytest.raises(ValueError):
            build_raw_zone_prefix(
                corpus="jsut",
                corpus_version="v1.0; DROP TABLE",
                domain="koe",
            )

    def test_invalid_domain(self):
        with pytest.raises(ValueError):
            build_raw_zone_prefix(
                corpus="jsut",
                corpus_version="v1.0",
                domain="/etc/passwd",
            )


# ════════════════════════════════════════════════════════════════════════
# 9. S3StorageBackend
# ════════════════════════════════════════════════════════════════════════


class TestS3StorageBackend:
    """Mock-based tests for S3StorageBackend protocol methods."""

    def test_satisfies_protocol(self):
        backend = S3StorageBackend(bucket="forge")
        assert isinstance(backend, StorageBackend)

    def test_init_validates_bucket(self):
        with pytest.raises(ValueError):
            S3StorageBackend(bucket="../escape")

    def test_prefix_stripped(self):
        backend = S3StorageBackend(bucket="forge", prefix="lake/raw/")
        assert backend._prefix == "lake/raw"

    @patch("modules.forge.storage.s3.upload_file")
    @patch("modules.forge.storage.s3.get_s3_client")
    def test_put(self, mock_get_client, mock_upload, tmp_path):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_upload.return_value = {"key": "pfx/test.wav"}

        backend = S3StorageBackend(bucket="forge", prefix="pfx")
        src = tmp_path / "test.wav"
        src.write_bytes(b"audio")

        result = backend.put("test.wav", src)
        assert result == "pfx/test.wav"
        mock_upload.assert_called_once()

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_get_to_path(self, mock_get_client, tmp_path):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        backend = S3StorageBackend(bucket="forge", prefix="pfx")
        dest = tmp_path / "out" / "test.wav"

        result = backend.get_to_path("test.wav", dest)
        assert result == dest
        mock_client.download_file.assert_called_once_with(
            "forge", "pfx/test.wav", str(dest)
        )
        # Parent dir should be created
        assert dest.parent.exists()

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_get_to_path_not_found(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Simulate 404 error
        error_response = {"Error": {"Code": "404"}}
        client_error = type("ClientError", (Exception,), {
            "response": error_response,
        })()
        # Make it an instance we can match
        mock_client.exceptions.ClientError = type(client_error)
        mock_client.download_file.side_effect = client_error

        backend = S3StorageBackend(bucket="forge")
        with pytest.raises(NotFoundError, match="Key not found"):
            backend.get_to_path("missing.wav", Path("/tmp/out.wav"))

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_exists_true(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        backend = S3StorageBackend(bucket="forge", prefix="pfx")
        assert backend.exists("test.wav") is True
        mock_client.head_object.assert_called_once_with(
            Bucket="forge", Key="pfx/test.wav"
        )

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_exists_false(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Simulate 404
        error_response = {"Error": {"Code": "404"}}
        client_error = type("ClientError", (Exception,), {
            "response": error_response,
        })()
        mock_client.exceptions.ClientError = type(client_error)
        mock_client.head_object.side_effect = client_error

        backend = S3StorageBackend(bucket="forge")
        assert backend.exists("missing.wav") is False

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_list_keys(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock paginator
        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [
                {"Key": "pfx/a.wav"},
                {"Key": "pfx/b.wav"},
            ]},
            {"Contents": [
                {"Key": "pfx/c.wav"},
            ]},
        ]

        backend = S3StorageBackend(bucket="forge", prefix="pfx")
        keys = list(backend.list_keys(""))

        assert keys == ["a.wav", "b.wav", "c.wav"]
        mock_client.get_paginator.assert_called_once_with("list_objects_v2")

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_list_keys_empty(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [{}]  # No Contents key

        backend = S3StorageBackend(bucket="forge", prefix="pfx")
        keys = list(backend.list_keys(""))
        assert keys == []

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_list_keys_returns_iterator(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_paginator = MagicMock()
        mock_client.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "a.wav"}]},
        ]

        backend = S3StorageBackend(bucket="forge")
        result = backend.list_keys("")
        assert isinstance(result, Iterator)

    @patch("modules.forge.storage.s3.get_s3_client")
    def test_no_client_raises(self, mock_get_client):
        mock_get_client.return_value = None

        backend = S3StorageBackend(bucket="forge")
        with pytest.raises(S3ConfigError, match="not configured"):
            backend.exists("test.wav")

    def test_full_key_no_prefix(self):
        backend = S3StorageBackend(bucket="forge")
        assert backend._full_key("test.wav") == "test.wav"

    def test_full_key_with_prefix(self):
        backend = S3StorageBackend(bucket="forge", prefix="lake/raw")
        assert backend._full_key("test.wav") == "lake/raw/test.wav"
