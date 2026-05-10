"""
Tasks module.

This module contains all asynchronous tasks for the application.
Import all task modules here to ensure they are discovered.
"""

from .telemetry import send_telemetry
from .logging import log_audio_task
from .scheme_scheduler import start_scheme_scheduler, stop_scheme_scheduler

__all__ = [
    'send_telemetry',
    'log_audio_task',
    'start_scheme_scheduler',
    'stop_scheme_scheduler',
]
