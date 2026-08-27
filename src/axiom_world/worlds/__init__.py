"""World implementations.

A *world* supplies the rule engine, the scenario generator, and the oracle that
the verifier scores against. Worlds are resolved by name so that adding one does
not require touching training or evaluation entry points.

Protocol v1 uses ``playworld`` (fully observable, open-loop).
Protocol v3.0-CLB adds ``playworld_po`` (partially observable, closed-loop),
which inherits v1's rule engine, action space, and legality semantics unchanged.
"""

from axiom_world.worlds.registry import (
    get_world,
    list_worlds,
    register_world,
)

__all__ = ["get_world", "list_worlds", "register_world"]
