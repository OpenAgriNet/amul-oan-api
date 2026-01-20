"""
Translation Class
- Handles translation of data structures while preserving non-string values -> using Depth First Search (DFS)
- Supports Google and Bhashini APIs.
- Can exclude specific keys from translation.
"""
import os
import re
import json
import requests
from dotenv import load_dotenv
from google.cloud import translate_v2 as translate
from google.oauth2 import service_account
from typing import Any, Dict, List, Tuple, Union, Set

load_dotenv(override=True)

term_pairs = json.load(open('assets/word_mapping_reduced_1000.json', 'r', encoding='utf-8'))

def fix_underscores(text):
    """Replace underscores with spaces -> underscores."""
    if isinstance(text, str):
        text = re.sub(r'[_] [_]', '__', text)
        text = re.sub(r'[_] [_]', '__', text)
    return text


def markdown_to_chunks(text):
    """Convert markdown to chunks (For Bhashini)"""
    # Only match the start of lines for headings and lists (excluding * as a list marker)
    pattern = r'(^\s*(?:#{1,6}|\-|\+|\d+\.)\s+)' ### This matches with heading and list markers at start of lines only (no asterisk)
    parts = re.split(pattern, text, flags=re.MULTILINE)

    chunks = []
    for part in parts:
        if not part:
            continue

        # Check if this is a heading/list marker (excluding * as a list marker)
        if re.match(r'^\s*(?:#{1,6}|\-|\+|\d+\.)\s+$', part):
            chunks.append({'start': part, 'text': '', 'end': ''})
        else:
            # Treat the entire part as regular text
            chunks.append({'start': '', 'text': part.strip(), 'end': '\n' if part.endswith('\n') else ''})

    return chunks


def chunks_to_markdown(chunks):
    """Convert a list of chunks (For Bhashini) back to markdown."""
    return ''.join(f"{c['start']}{c['text']}{c['end']}" for c in chunks)


def add_gujarati_terms(text: str, term_pairs=term_pairs) -> str:
    """
    Add Gujarati translations in brackets for matching English terms in the text.
    Example: "Click the button" -> "Click (ક્લિક) the button (બટન)"
    
    Args:
        text: English text to process
        term_pairs: List of dictionaries with 'en' and 'gu' keys for term pairs
        
    Returns:
        Text with Gujarati translations added in brackets for matching terms
    """
    if not text or not term_pairs:
        return text

    # Extract valid English-Gujarati pairs and sort by length
    valid_terms = []
    for term in term_pairs:
        en_term = term.get('en', '').strip()
        gu_term = term.get('gu', '').strip()
        
        if not en_term or not gu_term:
            continue
            
        # Normalize whitespace
        en_term = ' '.join(en_term.split())
        gu_term = ' '.join(gu_term.split())
        valid_terms.append((en_term, gu_term))
    
    # Sort by length (longest first) to handle overlapping terms
    valid_terms.sort(key=lambda x: len(x[0]), reverse=True)
    
    if not valid_terms:
        return text

    # Create pattern parts and mapping
    pattern_parts = []
    term_mapping = {}  # English lowercase -> Gujarati
    case_mapping = {}  # English lowercase -> English original case
    
    for en_term, gu_term in valid_terms:
        try:
            # Escape special regex characters
            escaped_term = re.escape(en_term)
            pattern_parts.append(escaped_term)
            
            # Store mappings
            term_mapping[en_term.lower()] = gu_term
            case_mapping[en_term.lower()] = en_term
        except Exception:
            continue

    if not pattern_parts:
        return text

    try:
        # Compile pattern with word boundaries
        pattern = r'(?:(?<=\s)|^)(?:' + '|'.join(pattern_parts) + r')(?=\s|$)'
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except re.error:
        # Fallback to simpler pattern
        pattern = r'\b(?:' + '|'.join(pattern_parts) + r')\b'
        compiled_pattern = re.compile(pattern, re.IGNORECASE)

    def replace_match(match):
        """Replace matched English term with "term (ગુજરાતી શબ્દ)" """
        term = match.group(0)
        lookup_term = term.strip().lower()
        
        gujarati = term_mapping.get(lookup_term, '')
        if gujarati:
            original_case = case_mapping.get(lookup_term, term)
            return f"{original_case} ({gujarati})"
        return term

    try:
        return compiled_pattern.sub(replace_match, text)
    except Exception:
        return text

class BaseTranslator:
    """Base translator class with common data handling methods."""
    def __init__(self, source_lang='en', target_lang='hi', batch_size=4, term_pairs=None):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size
        self.term_pairs = term_pairs or []
        self._compiled_pattern = None
        self._term_mapping = {}
        self._case_mapping = {}  # Maps lowercase to original case
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns and build term mapping for efficient replacements."""
        if not self.term_pairs:
            return

        # Filter valid terms and sort by length (longest first)
        valid_terms = []
        for term in self.term_pairs:
            source = term.get(self.source_lang, '').strip()
            target = term.get(self.target_lang, '').strip()
            
            # Skip invalid or empty terms
            if not source or not target:
                continue
                
            # Handle potential Unicode characters and normalize whitespace
            source = ' '.join(source.split())  # Normalize whitespace
            target = ' '.join(target.split())  # Normalize whitespace
            
            valid_terms.append((source, target))
            
        # Sort by length (longest first) to handle overlapping terms correctly
        valid_terms.sort(key=lambda x: len(x[0]), reverse=True)
        
        if not valid_terms:
            return

        # Create pattern parts and mapping
        pattern_parts = []
        for source, target in valid_terms:
            try:
                # Double escape backslashes first to handle Windows paths or similar
                source_escaped = source.replace('\\', '\\\\')
                # Then escape all regex special characters
                escaped_source = re.escape(source_escaped)
                pattern_parts.append(escaped_source)
                
                # Store lowercase version in term_mapping
                self._term_mapping[source.lower()] = target
                # Store original case in case_mapping
                self._case_mapping[source.lower()] = source
            except Exception as e:
                # Skip problematic patterns but continue processing others
                continue

        # Compile single pattern with word boundaries
        if pattern_parts:
            try:
                # Use lookaround assertions for better word boundary handling with case insensitive flag
                pattern = r'(?:(?<=\s)|^)(?:' + '|'.join(pattern_parts) + r')(?=\s|$)'
                self._compiled_pattern = re.compile(pattern, re.IGNORECASE)
            except re.error:
                # Fallback to simpler pattern if complex one fails
                pattern = r'\b(?:' + '|'.join(pattern_parts) + r')\b'
                self._compiled_pattern = re.compile(pattern, re.IGNORECASE)

    def _should_skip_path(self, path: List[Union[str, int]], exclude_keys: Set[str] = None) -> bool:
        """
        Determine if a path should be skipped based on excluded keys.
        """
        if not exclude_keys:
            return False
        return any(str(key) in exclude_keys for key in path)
        
    def _should_translate_string(self, text: str) -> bool:
        """
        Determine if a string should be translated based on its content.
        Skip strings that only contain whitespace or non-alphanumeric characters.
        """
        # Skip empty strings
        if not text or text.isspace():
            return False
            
        # Check if the string contains any alphanumeric characters
        return any(char.isalnum() for char in text)

    def _add_paired_translations(self, text: str) -> str:
        """Add translations in parentheses for dictionary terms using pre-compiled pattern."""
        if not self._compiled_pattern or not self._term_mapping:
            return text

        def replace_match(match):
            """Replace matched term with term (translation)"""
            term = match.group(0)
            # Handle potential leading/trailing whitespace
            lookup_term = term.strip().lower()
            
            translation = self._term_mapping.get(lookup_term, '')
            if translation:
                # Use original case if available, otherwise use matched case
                original_case = self._case_mapping.get(lookup_term, term)
                return f"{original_case} ({translation})"
            return term

        try:
            return self._compiled_pattern.sub(replace_match, text)
        except Exception:
            # Fallback to original text if replacement fails
            return text

    def translate_texts(self, texts: List[str]) -> List[str]:
        """Abstract method to be implemented by concrete translators."""
        raise NotImplementedError("translate_texts method must be implemented by subclasses.")

    def _collect_translatable_strings(self, data: Any, exclude_keys: Set[str] = None) -> List[Tuple[List[Union[str, int]], str]]:
        """First pass: Collect all translatable strings with their paths."""
        strings_to_translate = []
        
        def dfs(current_data: Any, current_path: List[Union[str, int]]) -> None:
            # Skip this branch if the current path contains an excluded key
            if self._should_skip_path(current_path, exclude_keys):
                return

            if isinstance(current_data, str):
                # Only add strings that should be translated
                if self._should_translate_string(current_data):
                    strings_to_translate.append((current_path.copy(), current_data))
            elif isinstance(current_data, dict):
                for key, value in current_data.items():
                    current_path.append(key)
                    dfs(value, current_path)
                    current_path.pop()
            elif isinstance(current_data, list):
                for idx, item in enumerate(current_data):
                    current_path.append(idx)
                    dfs(item, current_path)
                    current_path.pop()

        dfs(data, [])
        return strings_to_translate

    def _reconstruct_data(self, original_data: Any, translated_strings: List[str], 
                         paths: List[List[Union[str, int]]]) -> Any:
        """Second pass: Reconstruct the original structure with translated strings."""
        if not paths:  # If no paths, return original data unchanged
            return original_data

        # Create a deep copy of the original data
        result = self._deep_copy(original_data)
        
        # Create a mapping of translations - convert paths to tuples for hashing
        translations = dict(zip(map(tuple, paths), translated_strings))
        
        # Reconstruct the data structure
        for path, translated_text in translations.items():
            current = result
            for i, key in enumerate(path[:-1]):
                current = current[key]
            current[path[-1]] = translated_text
            
        return result

    def _deep_copy(self, data: Any) -> Any:
        """Create a deep copy of the data structure."""
        if isinstance(data, dict):
            return {k: self._deep_copy(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy(item) for item in data]
        else:
            return data

    def translate(self, data: Any, exclude_keys: Set[str] = None, use_term_pairs: bool = False) -> Any:
        """
        Translate any data structure while preserving structure and non-string values.
        Handles single strings, lists, dictionaries, and nested structures.
        
        Args:
            data: The data to translate
            exclude_keys: Set of keys to exclude from translation
            use_term_pairs: Whether to add paired translations before translation
        """
        # Handle simple string case directly
        if isinstance(data, str):
            if not exclude_keys and self._should_translate_string(data):
                text = self._add_paired_translations(data) if use_term_pairs else data
                return self.translate_texts([text])[0]
            return data
            
        # Collect all translatable strings with their paths
        paths_and_strings = self._collect_translatable_strings(data, exclude_keys)
        
        if not paths_and_strings:
            return data
            
        # Separate paths and strings
        paths, strings = zip(*paths_and_strings)
        
        # Apply paired translations if requested
        if use_term_pairs:
            strings = [self._add_paired_translations(text) for text in strings]
        
        # Batch translate all strings at once
        translated_strings = self.translate_texts(list(strings))
        
        # Reconstruct the data structure with translations
        return self._reconstruct_data(data, translated_strings, paths)


class BhashiniTranslator(BaseTranslator):
    """Translator implementation using the Bhashini API."""
    def __init__(self, source_lang='en', target_lang='hi', batch_size=4, term_pairs=None):
        super().__init__(source_lang, target_lang, batch_size, term_pairs)
        self.session = requests.Session()
        self.api_key = os.getenv("MEITY_API_KEY_VALUE")
        self.base_url = 'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'

    def translate_texts(self, texts: List[str]) -> List[str]:
        """Translate a list of texts using the Bhashini API."""
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        data = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        #"serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
                        "serviceId": "bhashini/ai4bharat/indictrans-v3",
                        "language": {
                            "sourceLanguage": self.source_lang,
                            "targetLanguage": self.target_lang
                        }
                    }
                }
            ],
            "inputData": {
                "input": [{"source": text} for text in texts]
            }
        }
        response = self.session.post(self.base_url, headers=headers, json=data, timeout=(10, 30))

        if response.status_code != 200:
            raise Exception(f"Error: {response.status_code} {response.text}")

        response_json = response.json()
        return [item['target'] for item in response_json['pipelineResponse'][0]['output']]


class GoogleTranslator(BaseTranslator):
    """Translator implementation using Google Translation API."""
    def __init__(self, source_lang='en', target_lang='hi', batch_size=4, term_pairs=None):
        super().__init__(source_lang, target_lang, batch_size, term_pairs)
        
        # Get the absolute path to the credentials file
        credentials_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'cloud-translate-credentials.json')
        
        if not os.path.exists(credentials_path):
            raise ValueError(f"Google Cloud credentials file not found at: {credentials_path}")
            
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize the client with explicit credentials
        self.client = translate.Client(credentials=credentials)

    def translate_texts(self, texts: List[str]) -> List[str]:
        """Translate a list of texts using the Google Translation API."""
        results = self.client.translate(
            texts,
            source_language=self.source_lang,
            target_language=self.target_lang,
        )
        if not results:
            raise Exception("Error: Translation failed")
        return [result['translatedText'] for result in results]