from dotenv import load_dotenv
from voice.helpers.utils import get_logger
from voice.app.config import settings
from shared.auth.deps import build_jwt_auth

load_dotenv()

logger = get_logger(__name__)

_bundle = build_jwt_auth(settings, logger)

oauth2_scheme = _bundle.oauth2_scheme
chat_oauth2_scheme = _bundle.chat_oauth2_scheme
public_key = _bundle.public_key
get_current_user = _bundle.get_current_user
get_chat_user = _bundle.get_chat_user
