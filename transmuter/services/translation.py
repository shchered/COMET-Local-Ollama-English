# -*- coding: utf-8 -*-
"""
Translation and language detection service module

Provides:
- _detect_language: lightweight language detection (Chinese/English/unknown)
- _contains_refusal_keywords: detect refusal-tone keywords
- _deepl_translate_to_zh: DeepL English→Chinese translation with exponential backoff retry

Notes:
- This module only uses DeepL for content translation and does not call an LLM for translation
"""

import re
import time
import random
import requests
from typing import Dict, Any

from config import (
    DEEPL_API_KEY, DEEPL_BASE_URL,
    TRANSLATION_RETRY_LIMIT, TRANSLATION_TIMEOUT,
    TRANSLATION_SPLIT_SENTENCES, TRANSLATION_PRESERVE_FORMATTING, TRANSLATION_FORMALITY
)


def _detect_language(text: str) -> str:
    try:
        if not isinstance(text, str):
            return 'unknown'
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        letters = sum(ch.isalpha() for ch in text)
        total = len([ch for ch in text if not ch.isspace()])
        ratio = (letters / total) if total > 0 else 0
        if ratio > 0.7:
            return 'en'
        return 'unknown'
    except Exception:
        return 'unknown'


_REFUSAL_KEYWORDS = [
    'i cannot', 'i can\'t', 'cannot help', 'cannot provide', 'refuse to', 'sorry',
    '\u65e0\u6cd5\u63d0\u4f9b', '\u4e0d\u80fd\u63d0\u4f9b', '\u4e0d\u80fd\u5e2e\u52a9', '\u62d2\u7edd', '\u62b1\u6b49'
]


def _contains_refusal_keywords(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    tl = text.lower()
    for kw in _REFUSAL_KEYWORDS:
        if kw in tl:
            return True
    return False


def _deepl_translate_to_zh(text: str) -> Dict[str, Any]:
    """Translate English to Chinese using the DeepL API, with exponential backoff retry."""
    if not DEEPL_API_KEY:
        return {
            'translated': text,
            'success': False,
            'did_translate': False,
            'retries': 0,
            'source_language': 'en',
            'error': 'DEEPL_API_KEY not set'
        }
    # Dynamically adjust the endpoint based on the key type to prevent configuration mismatches
    if str(DEEPL_API_KEY).endswith(":fx"):
        url = "https://api-free.deepl.com/v2/translate"
    else:
        url = "https://api.deepl.com/v2/translate"
    headers = { 'Authorization': f'DeepL-Auth-Key {DEEPL_API_KEY}' }
    data_base = {
        'source_lang': 'EN',
        'target_lang': 'ZH',
        'split_sentences': TRANSLATION_SPLIT_SENTENCES,
        'preserve_formatting': TRANSLATION_PRESERVE_FORMATTING,
        'formality': TRANSLATION_FORMALITY
    }
    last_error = None
    for attempt in range(TRANSLATION_RETRY_LIMIT):
        try:
            data = { **data_base, 'text': text }
            resp = requests.post(url, data=data, headers=headers, timeout=TRANSLATION_TIMEOUT)
            if resp.status_code == 200:
                js = resp.json()
                translations = js.get('translations') or []
                if translations:
                    translated_text = translations[0].get('text', text)
                    return {
                        'translated': translated_text,
                        'success': True,
                        'did_translate': True,
                        'retries': attempt,
                        'source_language': 'en'
                    }
                else:
                    last_error = 'No translations in response'
            elif resp.status_code in (429, 500, 502, 503, 504):
                last_error = f'HTTP {resp.status_code}'
            else:
                try:
                    err_msg = resp.text
                except Exception:
                    err_msg = f'HTTP {resp.status_code}'
                return {
                    'translated': text,
                    'success': False,
                    'did_translate': False,
                    'retries': attempt,
                    'source_language': 'en',
                    'error': err_msg
                }
        except Exception as e:
            last_error = str(e)
        sleep_time = (2 ** attempt) + random.random() * 0.1
        time.sleep(sleep_time)
    return {
        'translated': text,
        'success': False,
        'did_translate': False,
        'retries': TRANSLATION_RETRY_LIMIT,
        'source_language': 'en',
        'error': last_error or 'Translation failed'
    }