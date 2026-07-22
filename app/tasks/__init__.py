"""
Tasks module.

This module contains all asynchronous tasks for the application.
Import all task modules here to ensure they are discovered.
"""

from .suggestions import create_suggestions
from .scheme_scheduler import start_scheme_scheduler, stop_scheme_scheduler

__all__ = [
    'create_suggestions',
    'start_scheme_scheduler',
    'stop_scheme_scheduler',
]
