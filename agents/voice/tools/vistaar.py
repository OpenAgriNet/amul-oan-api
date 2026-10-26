"""
Bharat Vistaar weather and mandi prices on a call (AMUL-81).

These are chat's tools from agents/tools/vistaar.py, reused unchanged: same
FarmerContext, same location resolution (named place → session → the farmer's
district → Anand), same output. Voice adds one thing, a time cap. One Vistaar
request may take AMUL_NETWORK_TIMEOUT_S (35 s), and a zero-row answer walks up
to three nearby places in turn, while a voice turn that passes the 60 s ingress
window drops the call. Past VOICE_VISTAAR_BUDGET_S the caller hears the tool's
own "temporarily unavailable" line and the call stays up.
"""
import asyncio
import functools

from agents.tools import vistaar
from app.config import settings
from helpers.utils import get_logger

logger = get_logger(__name__)


def _within_voice_budget(func, unavailable: str):
    """Run a Vistaar tool under the voice time cap, keeping its name,
    signature and docstring so the model sees the same tool chat does."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=settings.voice_vistaar_budget_s)
        except asyncio.TimeoutError:
            logger.warning(
                "voice %s exceeded its %.0f s budget", func.__name__, settings.voice_vistaar_budget_s
            )
            return unavailable
    return wrapper


get_vistaar_weather = _within_voice_budget(
    vistaar.get_vistaar_weather,
    "Weather is temporarily unavailable from Bharat Vistaar.",
)
get_vistaar_mandi_prices = _within_voice_budget(
    vistaar.get_vistaar_mandi_prices,
    "Mandi prices are temporarily unavailable from Bharat Vistaar. Please try again shortly.",
)
