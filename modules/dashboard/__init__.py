"""
KOE Training Dashboard.

FastAPI backend for monitoring training runs.
"""

from .backend import create_app

__all__ = ["create_app"]
