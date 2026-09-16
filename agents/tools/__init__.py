"""Agent tool package.

The registry is loaded lazily so importing a model or Beckn adapter does not
construct every Pydantic-AI tool and create circular imports.
"""


def __getattr__(name: str):
    if name == "TOOLS":
        from agents.tools.registry import TOOLS

        return TOOLS
    raise AttributeError(name)


__all__ = ["TOOLS"]
