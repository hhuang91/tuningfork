"""Content-addressed blob storage.

Bytes are stored once under the hex sha256 of their content, so identical messages across
runs and forks cost nothing extra. This is the one Phase 0 storage component that carries
forward unchanged into the Phase 1 message tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

#: How recently a blob must have been written or seen again for a sweep to leave it, though
#: nothing refers to it yet: a recording in flight writes its blobs before the rows that
#: refer to them.
GRACE = 120.0


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / digest[2:]

    def put(self, data: bytes) -> str:
        """Store `data`, returning its digest. Idempotent. A blob already stored is marked
        as seen now, so a sweep running meanwhile leaves it (see `sweep`)."""
        digest = hashlib.sha256(data).hexdigest()
        path = self._path(digest)
        try:
            os.utime(path)
        except FileNotFoundError:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a concurrent reader never observes a partial blob, and two
            # writers racing on the same content cannot corrupt each other.
            tmp = path.parent / f"{path.name}.{uuid.uuid4().hex}.tmp"
            tmp.write_bytes(data)
            tmp.replace(path)
        return digest

    def put_json(self, obj: Any) -> str:
        """Store a JSON value with its keys in the order they came in.

        Not sorted: an inference server renders a request's tool schemas and past tool-call
        arguments in the order it receives them, so a request rebuilt with sorted keys is a
        different prompt from the one recorded. Identity in the message tree has its own
        canonical hash (`tree.exact_hash`), so it does not depend on this order.
        """
        return self.put(json.dumps(obj, ensure_ascii=False).encode())

    def get(self, digest: str) -> bytes:
        return self._path(digest).read_bytes()

    def get_json(self, digest: str) -> Any:
        return json.loads(self.get(digest))

    def __contains__(self, digest: object) -> bool:
        return isinstance(digest, str) and self._path(digest).exists()

    def sweep(self, live: set[str], *, grace: float = GRACE, held: list[str] | None = None) -> tuple[int, int]:
        """Delete every blob not in `live` that was not written or seen within `grace`
        seconds, and any write left half-done that long. Returns (blobs deleted, bytes); each
        kept only for the grace is added to `held`."""
        cutoff = time.time() - grace
        deleted = freed = 0
        for directory in self.root.iterdir() if self.root.is_dir() else ():
            if not directory.is_dir() or len(directory.name) != 2:
                continue
            for path in directory.iterdir():
                digest = directory.name + path.name
                if not path.name.endswith(".tmp") and digest in live:
                    continue
                try:
                    found = path.stat()
                    if found.st_mtime > cutoff:
                        if held is not None:
                            held.append(digest)
                        continue
                    path.unlink()
                except FileNotFoundError:
                    continue
                deleted += 1
                freed += found.st_size
        return deleted, freed
