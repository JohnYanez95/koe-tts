"""Cache management — atomic extraction with manifest tracking.

Public API
----------
- ``CacheError`` — base exception for cache operations
- ``CacheManager`` — atomic download + extract + manifest
- ``CacheManifest`` — manifest dataclass
- ``ManifestEntry`` — single file entry in a manifest
"""

from modules.forge.cache.manager import CacheError, CacheManager
from modules.forge.cache.manifest import CacheManifest, ManifestEntry

__all__ = ["CacheError", "CacheManager", "CacheManifest", "ManifestEntry"]
