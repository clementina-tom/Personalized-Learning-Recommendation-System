"""
plrs.api.auth
=============
API key management — see full implementation in the complete repo zip.
"""
import secrets, time, hashlib, json, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TIERS = {
    "free":     {"requests_per_minute": 10,    "requests_per_day": 100},
    "standard": {"requests_per_minute": 60,    "requests_per_day": 1_000},
    "premium":  {"requests_per_minute": 300,   "requests_per_day": 10_000},
    "internal": {"requests_per_minute": 10_000,"requests_per_day": 1_000_000},
}
DEFAULT_TIER = "standard"

@dataclass
class APIKey:
    key_id: str; name: str; tier: str; created_at: float
    is_active: bool = True; metadata: dict = field(default_factory=dict)
    @property
    def limits(self): return TIERS.get(self.tier, TIERS[DEFAULT_TIER])
    @property
    def requests_per_minute(self): return self.limits["requests_per_minute"]
    @property
    def requests_per_day(self): return self.limits["requests_per_day"]

class KeyStore:
    KEY_PREFIX = "plrs_"; KEY_BYTES = 16
    def __init__(self, persist_path=None):
        self._keys = {}
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists(): self._load()
    def create_key(self, name, tier=DEFAULT_TIER, metadata=None):
        if tier not in TIERS: raise ValueError(f"Unknown tier '{tier}'")
        raw = self.KEY_PREFIX + secrets.token_hex(self.KEY_BYTES)
        self._keys[raw] = APIKey(key_id=raw, name=name, tier=tier, created_at=time.time(), metadata=metadata or {})
        if self._persist_path: self._save()
        return raw
    def validate(self, raw):
        k = self._keys.get(raw)
        if k is None: raise KeyError("API key not found.")
        if not k.is_active: raise ValueError(f"Key '{k.name}' is inactive.")
        return k
    def revoke(self, raw):
        k = self._keys.get(raw)
        if k is None: raise KeyError("API key not found.")
        k.is_active = False
        if self._persist_path: self._save()
    def delete(self, raw):
        if raw not in self._keys: raise KeyError("API key not found.")
        del self._keys[raw]
        if self._persist_path: self._save()
    def list_keys(self):
        return [{"name": k.name, "tier": k.tier, "is_active": k.is_active, "created_at": k.created_at, "limits": k.limits, "key_prefix": k.key_id[:12] + "..."} for k in self._keys.values()]
    def __len__(self): return len(self._keys)
    def _save(self):
        import dataclasses
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "w") as f: json.dump({k: dataclasses.asdict(v) for k,v in self._keys.items()}, f, indent=2)
    def _load(self):
        with open(self._persist_path) as f: data = json.load(f)
        for k, d in data.items(): self._keys[k] = APIKey(**d)

_key_store = KeyStore(persist_path=os.getenv("PLRS_KEYS_PATH"))
def get_key_store(): return _key_store
def init_key_store(persist_path=None):
    global _key_store; _key_store = KeyStore(persist_path=persist_path); return _key_store
