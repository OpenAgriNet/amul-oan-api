"""The channel seam: one orchestrator, populated differently per surface.

See ``docs/channel-seam-design.md``. ``app.turn.types`` holds the transport-free
contract (``Turn``, ``Emission``, ``SurfaceProfile``); the chat surface's
population of it and ``run_turn`` itself live in ``app.services.chat``.
"""
