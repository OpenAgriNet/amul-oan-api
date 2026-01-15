# Import all routers to make them available when importing from app.routers
# This allows main.py to do: from app.routers import chat, transcribe, suggestions, tts, auth
from . import chat
from . import transcribe
from . import suggestions
from . import tts
from . import health
from . import auth