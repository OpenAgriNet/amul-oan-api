# sva/helpers/utils.py

import os
import re
from typing import List, Dict
import logging
from dotenv import load_dotenv
import base64
import tiktoken
import unicodedata as ud
from datetime import datetime
import simplejson as json
from jinja2 import Environment, FileSystemLoader
import pytz

load_dotenv()

def get_today_date_str() -> str:
    """Get today's date as a string in the format Monday, 23rd May 2025."""
    ist = pytz.timezone('Asia/Kolkata')
    today = datetime.now(ist)
    return today.strftime('%A, %d %B %Y')


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



def get_prompt(prompt_file: str, context: Dict = {}, prompt_dir: str = "assets/prompts") -> str:
    """Load a prompt from a file and format it with a context using Jinja2 templating.

    Args:
        prompt_file (str): Name of the prompt file.
        context (dict, optional): Context to format the prompt with. Defaults to {}.
        prompt_dir (str, optional): Path to the prompt directory. Defaults to 'assets/prompts'.

    Returns:
        str: prompt
    """
    # if extension is not .md, add it
    if not prompt_file.endswith(".md"):
        prompt_file += ".md"

    # Create Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(prompt_dir),
        autoescape=False  # We don't want HTML escaping for our prompts
    )

    # Get the template
    template = env.get_template(prompt_file)

    # Render the template with the context
    prompt = template.render(**context) if context else template.render()
    
    return prompt

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