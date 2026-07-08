# sva/helpers/utils.py

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import logging
import boto3
from dotenv import load_dotenv
import base64
import tiktoken
import unicodedata as ud
from datetime import datetime
import simplejson as json
from jinja2 import Environment, FileSystemLoader, Template
import pytz
from helpers.gujarati_numbers import normalize_numbers_for_tts
from app.models.farmer import FarmerModel
from app.models.union import UnionName

load_dotenv()

# In-memory prompt template cache (populated at app startup; no disk I/O at request time)
PROMPT_TEMPLATES_CACHE: Dict[str, str] = {}


def get_s3_client():
    """Get S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION'),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL", None)
    )


def get_today_date_str() -> str:
    """Get today's date as a string in the format Monday, 23rd May 2025."""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist)
    return today.strftime('%A, %d %B %Y')


def get_today_datetime_str() -> str:
    """Get current date and time as a string in Asia/Kolkata."""
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    return now.strftime('%A, %d %B %Y %I:%M %p %Z')


def get_logger(name):
    """Get logger object."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def count_tokens_str(doc: str) -> int:
    """Count tokens in a string.

    Args:
        doc (str): String to count tokens for.
    Returns:
        int: number of tokens in the string

    """
    encoder = tiktoken.get_encoding('cl100k_base')
    return len(encoder.encode(doc, disallowed_special=()))


def count_tokens_for_part(part) -> int:
    """Count tokens for a message part, handling different part types appropriately.
    
    Args:
        part: A message part (TextPart, ToolCallPart, etc.)
    Returns:
        int: number of tokens in the part
    """
    if hasattr(part, 'content'):
        return count_tokens_str(str(part.content))
    elif hasattr(part, 'part_kind') and part.part_kind == 'tool-call':
        # For tool calls, create a string representation of the tool name and args
        tool_str = f"tool: {part.tool_name}, args: {json.dumps(part.args)}"
        return count_tokens_str(tool_str)
    elif hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
        # For tool returns, use the result content
        return count_tokens_str(str(part.content))
    else:
        # For unknown part types, return 0 tokens
        return 0



def is_sentence_complete(text: str) -> bool:
    """Check if the text is a complete sentence.
    
    Args:
        text (str): Text to check.

    Returns:
        bool: True if the text is a complete sentence, False otherwise.
    """
    # Check if text ends with a sentence terminator (., !, ?) possibly followed by whitespace or newlines
    return text.endswith('\n')

def split_text(text: str) -> List[str]:
    """Split text into chunks based on newlines.
    
    Args:
        text (str): Text to split.

    Returns:
        list: List of chunks, split by newlines.
    """
    # Split on newlines and filter out empty strings
    chunks = [chunk + "\n" for chunk in text.split('\n')]
    return chunks


def remove_redundant_parenthetical(text: str) -> str:
    """
    Collapse "X (X)" → "X" for any Unicode text.

    * Works with Devanagari and other non-Latin scripts.
    * Keeps bullets, punctuation, spacing, etc. unchanged.
    * Normalises both copies of the term to NFC first so that
      visually-identical strings made of different code-point
      sequences (e.g., decomposed vowel signs) are still caught.
    """
    # Optional but helps when the same glyph can be encoded two ways
    text = ud.normalize("NFC", text)

    pattern = re.compile(
        r'''
        (?P<term>                 # 1st copy
            [^\s()]+              #   – at least one non-space, non-paren char
            (?:\s+[^\s()]+)*      #   – then zero-or-more <space + word>
        )
        \s*                       # spaces before '('
        \(\s*
        (?P=term)                 # identical 2nd copy
        \s*\)                     # closing ')'
        ''',
        flags=re.UNICODE | re.VERBOSE,
    )

    return pattern.sub(lambda m: m.group('term'), text)

def remove_redundant_angle_brackets(text: str) -> str:
    """
    Collapse "X <X>" → "X" for any Unicode text.

    * Works with Devanagari and other non-Latin scripts.
    * Keeps bullets, punctuation, spacing, etc. unchanged.
    * Normalises both copies of the term to NFC first so that
      visually-identical strings made of different code-point
      sequences (e.g., decomposed vowel signs) are still caught.
    """
    # Optional but helps when the same glyph can be encoded two ways
    text = ud.normalize("NFC", text)

    pattern = re.compile(
        r'''
        (?P<term>                 # 1st copy
            [^\s<>]+              #   – at least one non-space, non-angle-bracket char
            (?:\s+[^\s<>]+)*      #   – then zero-or-more <space + word>
        )
        \s*                       # spaces before '<'
        <\s*
        (?P=term)                 # identical 2nd copy
        \s*>                      # closing '>'
        ''',
        flags=re.UNICODE | re.VERBOSE,
    )

    return pattern.sub(lambda m: m.group('term'), text)

def remove_redundant_square_brackets(text: str) -> str:
    """Collapse "X [X]" → "X" for any Unicode text."""
    text = ud.normalize("NFC", text)

    pattern = re.compile(
        r'''
        (?P<term>
            [^\s\[\]]+
            (?:\s+[^\s\[\]]+)*
        )
        \s*\[\s*
        (?P=term)
        \s*\]
        ''',
        flags=re.UNICODE | re.VERBOSE,
    )

    return pattern.sub(lambda m: m.group('term'), text)

def _replace_voice_abbreviations(text: str, lang_code: str) -> str:
    """Expand abbreviations that are awkward for voice output."""
    lang = (lang_code or "").strip().lower()

    if lang == "gu":
        replacements = [
            (r"(?i)\bkg\b", "કિલોગ્રામ"),
            (r"(?i)\bkgs\b", "કિલોગ્રામ"),
            (r"(?i)\bml\b", "મિલીલીટર"),
            (r"(?i)\bl\b", "લિટર"),
            (r"(?i)\bkm\b", "કિલોમીટર"),
            (r"(?i)\bcm\b", "સેન્ટીમીટર"),
            (r"(?i)કિ\.?\s*ગ્રા\.?", "કિલોગ્રામ"),
            (r"(?i)(?<!\w)ગ્રા\.?(?!\w)", "ગ્રામ"),
            (r"(?i)મિ\.?\s*લી\.?", "મિલીલીટર"),
            (r"(?i)%", " ટકા"),
            (r"(?i)°\s*c", " ડિગ્રી સેલ્સિયસ"),
        ]
    else:
        replacements = [
            (r"(?i)\bkg\b", "kilograms"),
            (r"(?i)\bkgs\b", "kilograms"),
            (r"(?i)\bml\b", "milliliters"),
            (r"(?i)\bl\b", "liters"),
            (r"(?i)\bkm\b", "kilometers"),
            (r"(?i)\bcm\b", "centimeters"),
            (r"(?i)%", " percent"),
            (r"(?i)°\s*c", " degrees Celsius"),
        ]

    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out)
    return out


def _remove_quantity_placeholders(text: str) -> str:
    """Remove placeholder dashes used as fake quantities before units."""
    out = text
    unit_words = (
        r"કિલોગ્રામ|ગ્રામ|મિલીલીટર|લિટર|"
        r"kilograms?|grams?|milliliters?|liters?"
    )
    out = re.sub(
        rf"([:：]\s*)(?:[-–—]{{1,3}})(?=\s*(?:{unit_words})\b)",
        r"\1",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"\s(?:--|––|——)\s", " ", out)
    return out

def normalize_voice_output(
    text: str,
    lang_code: str | None,
    *,
    replace_slash: bool = True,
    streaming: bool = False,
) -> str:
    """Normalize model output for voice playback before language filtering."""
    if not text:
        return text

    lang = (lang_code or "").strip().lower()
    out = ud.normalize("NFC", text)

    if lang == "gu":
        out = normalize_numbers_for_tts(out)

    if streaming:
        out = _replace_voice_abbreviations(out, lang)
        out = _remove_quantity_placeholders(out)
        out = re.sub(r"\.{3,}", ".", out)
        out = re.sub(r"([!?])\1+", r"\1", out)
        out = re.sub(r"([,;:])\1+", r"\1", out)
        if replace_slash:
            if lang == "gu":
                out = out.replace("/", " અથવા ")
            else:
                out = out.replace("/", " or ")
        return out

    # Strip markdown and structural noise that should never be spoken.
    out = out.replace("**", "").replace("__", "").replace("`", "").replace("~", "")
    out = re.sub(r"(?m)^\s*#+\s*", "", out)
    # Extended list-scrub: Latin bullets/dashes, em/en-dashes, Gujarati-style
    # bullets (•·), and Gujarati-numeral list markers (e.g. "૧. ", "૧) ").
    out = re.sub(r"(?m)^\s*(?:[-–—*•·]+|\d+[.)]|[૦-૯]+[.)]\s)\s*", "", out)

    out = _replace_voice_abbreviations(out, lang)
    out = _remove_quantity_placeholders(out)

    # Normalize common punctuation clutter.
    out = re.sub(r"\.{3,}", ".", out)
    out = re.sub(r"([!?])\1+", r"\1", out)
    out = re.sub(r"([,;:])\1+", r"\1", out)
    out = re.sub(r"\s*([,;:!?])\s*", r"\1 ", out)

    # Replace slash-separated alternatives with spoken language.
    if replace_slash:
        if lang == "gu":
            out = out.replace("/", " અથવા ")
        else:
            out = out.replace("/", " or ")

    # Collapse repeated bracketed copies before removing remaining brackets.
    out = remove_redundant_parenthetical(out)
    out = remove_redundant_square_brackets(out)
    out = remove_redundant_angle_brackets(out)

    # Brackets are visual syntax, not voice content.
    out = re.sub(r"[\[\]{}()<>]", " ", out)

    # Flatten line breaks into spoken text.
    out = re.sub(r"\s*\n+\s*", " ", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out

def post_process_translation(translation: str) -> str:
    """Post process translation.
    
    Args:
        translation (str): Translation to post process.

    Returns:
        str: Post processed translation.
    """
    # 1. Remove trailing `:` from text from each line
    lines = translation.split('\n')
    processed_lines = [line.rstrip(':') for line in lines]
    translation = '\n'.join(processed_lines)    
    # 2. Remove redundant parentheticals.
    translation = remove_redundant_parenthetical(translation)
    # 3. Remove redundant angle brackets.
    translation = remove_redundant_angle_brackets(translation)
    # 4. Remove double `::`
    translation = re.sub(r'::', ':', translation)
    translation = translation.replace(':**:', ':**')
    return translation



def load_prompt_templates(prompt_dir: Path) -> None:
    """Load all .md prompt templates from the given directory into PROMPT_TEMPLATES_CACHE.
    Call once at app startup (e.g. in FastAPI lifespan) to avoid disk I/O at request time.
    """
    prompt_dir = Path(prompt_dir)
    if not prompt_dir.is_dir():
        return
    for path in prompt_dir.glob("*.md"):
        key = path.stem  # e.g. voice_system_gu
        try:
            PROMPT_TEMPLATES_CACHE[key] = path.read_text(encoding="utf-8")
        except Exception as e:
            _log = logging.getLogger(__name__)
            _log.warning("Failed to load prompt template %s: %s", path, e)


def get_prompt(prompt_file: str, context: Dict = {}, prompt_dir: str = "assets/prompts") -> str:
    """Load a prompt from in-memory cache (if populated at startup) or disk, and render with context.

    Args:
        prompt_file (str): Name of the prompt file (with or without .md).
        context (dict, optional): Context to format the prompt with. Defaults to {}.
        prompt_dir (str, optional): Path to the prompt directory (used only if not in cache).

    Returns:
        str: Rendered prompt.
    """
    if not prompt_file.endswith(".md"):
        prompt_file += ".md"
    cache_key = Path(prompt_file).stem  # e.g. voice_system_gu

    if cache_key in PROMPT_TEMPLATES_CACHE:
        template = Template(PROMPT_TEMPLATES_CACHE[cache_key], autoescape=False)
        return template.render(**context) if context else template.render()

    # Fallback: load from disk (e.g. tests or if startup load was skipped)
    env = Environment(
        loader=FileSystemLoader(prompt_dir),
        autoescape=False,
    )
    template = env.get_template(prompt_file)
    return template.render(**context) if context else template.render()


def clean_output_by_language(text: str, lang_code: str | None) -> str:
    """Filter model output based on language.

    - Always allow whitespace.
    - Always allow basic sentence/word punctuation (.,!? and similar) for all languages.
    - For Gujarati (lang_code 'gu'), additionally restrict letters to the Gujarati Unicode block
      U+0A80..U+0AFF; everything else (Latin letters, other scripts) is stripped.
    """
    if not text:
        return text

    text = normalize_voice_output(text, lang_code)

    lang = (lang_code or "").strip().lower()
    # Basic punctuation to always allow
    allowed_punct = set(".!?,;:()[]{}\"'“”‘’-–—…")

    def _allowed(ch: str) -> bool:
        if ch.isspace():
            return True
        if ch in allowed_punct:
            return True
        code = ord(ch)
        if lang == "gu":
            # Gujarati block U+0A80..U+0AFF (includes letters, digits, signs)
            return 0x0A80 <= code <= 0x0AFF
        # For non-Gujarati, don't restrict characters beyond punctuation/whitespace
        return True

    return "".join(ch for ch in text if _allowed(ch))

def upload_audio_to_s3(audio_base64: str, session_id: str, bucket_name: str | None = None) -> Dict:
    """Upload base64 encoded audio to S3.
    
    Args:
        audio_base64 (str): Base64 encoded audio content
        session_id (str): Session ID for the conversation
        bucket_name (str, optional): S3 bucket name. Defaults to env variable.
        
    Returns:
        dict: Dictionary containing upload details
    """
    try:
        if not bucket_name:
            bucket_name = os.getenv('AWS_S3_BUCKET')
            
        if not bucket_name:
            raise ValueError("S3 bucket name not provided")
            
        # Decode base64 content
        audio_content = base64.b64decode(audio_base64)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"audio/{session_id}/{timestamp}.wav"
        
        # Get S3 client and upload
        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=audio_content,
            ContentType='audio/wav'
        )
        
        return {
            'status': 'success',
            'bucket': bucket_name,
            'key': filename,
            'session_id': session_id
        }
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error uploading audio to S3: {str(e)}")
        raise


def is_from_union(records: list[FarmerModel], union_name: UnionName) -> bool:
    return any(record.union_name == union_name.value for record in records)
