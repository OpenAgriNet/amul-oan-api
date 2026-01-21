"""
Tasks module.

This module contains all asynchronous tasks for the application.
Import all task modules here to ensure they are discovered.
"""

from .suggestions import create_suggestions

__all__ = [
    'create_suggestions', 
] 