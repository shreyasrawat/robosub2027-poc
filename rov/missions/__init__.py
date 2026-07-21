"""Mission scripts.

Each module here exposes a ``run(rov)`` function taking a connected
:class:`~rov.rov.ROV`. Keeping missions as plain functions rather than
classes means ``main.py`` can select one by name and a test can call it with
a stub.
"""

from . import square, gate

#: Missions selectable from the command line.
REGISTRY = {
    "square": square.run,
    "gate": gate.run,
}

__all__ = ["REGISTRY", "square", "gate"]
