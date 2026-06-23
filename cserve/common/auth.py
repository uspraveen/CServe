"""Authentication & API key management for CServe.

Design:
  - Every API request must include a Bearer token: `Authorization: Bearer csk_...`
  - Keys are stored hashed (SHA-256) so a DB leak doesn't expose secrets.
  - The raw key is only ever returned once — at creation time.
  - Each key belongs to a user_id and has:
      - A human-readable name (e.g. "production-backend")
      - Per-key rate limit (requests/min), 0 = unlimited
      - An is_admin flag (admins can create/revoke keys, see all users)
      - An enabled flag (soft-disable without deletion)
  - A bootstrap admin key is auto-created on first startup if no keys exist.

Key format: csk_{32 hex chars}  (CServe Key)
"""

from __future__ import annotations

import hashlib
import secrets
import time
from enum import StrEnum

from pydantic import BaseModel, Field

KEY_PREFIX = "csk_"
KEY_BYTES = 32


class KeyRole(StrEnum):
    USER = "user"
    ADMIN = "admin"


class ApiKey(BaseModel):
    """Stored representation of an API key (hash only, never the raw secret)."""
    key_id: str                          # short unique id for references
    key_hash: str                        # SHA-256 of the full key
    key_prefix: str = ""                 # first 8 chars for identification (csk_xxxx...)
    user_id: str
    name: str = ""                       # human-readable label
    role: KeyRole = KeyRole.USER
    rate_limit_rpm: int = 0              # 0 = unlimited
    enabled: bool = True
    created_at: float = Field(default_factory=time.time)
    last_used_at: float = 0.0
    total_requests: int = 0


class AuthenticatedUser(BaseModel):
    """Attached to each authenticated request — lightweight, no secrets."""
    key_id: str
    user_id: str
    role: KeyRole
    rate_limit_rpm: int


def generate_api_key() -> tuple[str, str, str]:
    """Generate a new API key.

    Returns:
        (raw_key, key_hash, key_id)
        raw_key is returned to the user ONCE and never stored.
    """
    raw = KEY_PREFIX + secrets.token_hex(KEY_BYTES)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    key_id = secrets.token_hex(6)
    return raw, key_hash, key_id


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()
