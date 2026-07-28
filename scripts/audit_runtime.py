#!/usr/bin/env python
"""Print the environment manifest (protocol §3). Run first in every session."""
from __future__ import annotations

import json

from axiom_world.runtime.audit import collect_environment_manifest

if __name__ == "__main__":
    print(json.dumps(collect_environment_manifest(), indent=2, sort_keys=True))
