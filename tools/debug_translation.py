# -*- coding: utf-8 -*-
"""
DeepL translation debugging script.

Purpose:
- Read queries from a file and run the SearchAttackApplication pipeline
- Auto English→Chinese translation (DeepL), print translation_info and summary
- Support raw mode: call _deepl_translate_to_zh directly to diagnose API issues

Examples:
- python tools/debug_translation.py --file data/instructions.txt --limit 5
- python tools/debug_translation.py --raw --file data/instructions.txt --limit 5
- python tools/debug_translation.py --provider deepl --file data/instructions.txt
"""

import os
import sys
import time
import json
import argparse
import logging
import importlib.util
from pathlib import Path


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def mask_key(key: str) -> str:
    if not key:
        return '(empty)'
    if len(key) <= 8:
        return key[:2] + '***' + key[-2:]
    return key[:4] + '***' + key[-4:]


def read_queries_from_file(path: Path, limit: int, encoding: str = 'utf-8'):
    queries = []
    with path.open('r', encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            queries.append(s)
            if len(queries) >= limit:
                break
    return queries


def print_result(idx: int, query: str, result: dict, show_translated: bool = True):
    # Extract translation_info
    ti = result.get('translation_info') or {}
    src_lang = ti.get('source_language')
    did = ti.get('did_translate')
    retries = ti.get('retries')
    success = ti.get('success')
    error = ti.get('error')
    translated = ti.get('translated')

    processed_query = result.get('processed_query')
    rewritten = result.get('query_rewritten')
    ok = result.get('success')

    def trunc(s, n=120):
        if s is None:
            return ''
        s = str(s)
        return s if len(s) <= n else s[:n] + '...'

    print(f"\n[{idx}] ok={ok} src_lang={src_lang} did_translate={did} retries={retries} success={success}")
    if error:
        print(f"    error: {error}")
    print(f"    original: {trunc(query)}")
    if show_translated and did:
        print(f"    translated: {trunc(translated)}")
    print(f"    processed: {trunc(processed_query)} rewritten={rewritten}")


def main():
    parser = argparse.ArgumentParser(description='DeepL translation and pipeline debug script')
    parser.add_argument('--file', type=str, default='data/instructions.txt', help='Input file path')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of lines to read')
    parser.add_argument('--raw', action='store_true', help='Raw mode: call _deepl_translate_to_zh directly')
    parser.add_argument('--provider', type=str, default='config', choices=['config','deepl','llm','auto'], help='Override translation provider (default: config)')
    parser.add_argument('--show-translated', action='store_true', help='Print a preview of the translation')
    parser.add_argument('--sleep', type=float, default=0.0, help='Seconds between requests')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding')
    parser.add_argument('--verbose', action='store_true', help='Show debug logs')
    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger('debug_translation')

    # Safely load the local config.py to avoid clashing with an external package of the same name
    project_root = Path(__file__).resolve().parents[1]
    # Ensure sibling packages (transmuter, etc.) are importable
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    config_path = project_root / 'config.py'
    spec = importlib.util.spec_from_file_location('config', str(config_path))
    config = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, 'Could not load local config.py'
    spec.loader.exec_module(config)
    # Ensure downstream modules all import the local config
    sys.modules['config'] = config

    # Show the current configuration
    log.info(f"config.TRANSLATION_PROVIDER={getattr(config, 'TRANSLATION_PROVIDER', None)}")
    log.info(f"config.DEEPL_BASE_URL={getattr(config, 'DEEPL_BASE_URL', None)}")
    log.info(f"config.DEEPL_API_KEY={mask_key(getattr(config, 'DEEPL_API_KEY', ''))}")
    log.info(f"retry_limit={getattr(config, 'TRANSLATION_RETRY_LIMIT', None)} timeout={getattr(config, 'TRANSLATION_TIMEOUT', None)}")
    log.info(f"split_sentences={getattr(config, 'TRANSLATION_SPLIT_SENTENCES', None)} preserve_formatting={getattr(config, 'TRANSLATION_PRESERVE_FORMATTING', None)} formality={getattr(config, 'TRANSLATION_FORMALITY', None)}")

    # Import the main application module only in non-Raw mode to avoid triggering LLM initialization
    tm = None
    if not args.raw:
        import transmuter.main as tm
        if args.provider != 'config':
            tm.TRANSLATION_PROVIDER = args.provider
            log.info(f"override tm.TRANSLATION_PROVIDER={tm.TRANSLATION_PROVIDER}")

    queries = read_queries_from_file(Path(args.file), args.limit, args.encoding)
    if not queries:
        log.error(f"File has no valid content: {args.file}")
        sys.exit(1)

    if args.raw:
        # Raw mode: detect the language and call the DeepL translation function directly (load by file path to avoid triggering package __init__)
        translation_path = project_root / 'transmuter' / 'services' / 'translation.py'
        tspec = importlib.util.spec_from_file_location('translation_service', str(translation_path))
        tmod = importlib.util.module_from_spec(tspec)
        assert tspec and tspec.loader, 'Could not load translation service module'
        tspec.loader.exec_module(tmod)
        _detect_language = getattr(tmod, '_detect_language')
        _deepl_translate_to_zh = getattr(tmod, '_deepl_translate_to_zh')
        for i, q in enumerate(queries, 1):
            lang = _detect_language(q)
            if lang == 'en':
                ti = _deepl_translate_to_zh(q)
                result = {
                    'success': ti.get('success', False),
                    'processed_query': ti.get('translated', q),
                    'query_rewritten': False,
                    'translation_info': ti
                }
            else:
                # Non-English: pass through without translation
                result = {
                    'success': True,
                    'processed_query': q,
                    'query_rewritten': False,
                    'translation_info': {
                        'did_translate': False,
                        'source_language': lang,
                        'retries': 0,
                        'success': True,
                        'translated': q
                    }
                }
            print_result(i, q, result, show_translated=args.show_translated)
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        # Application pipeline mode
        app = tm.SearchAttackApplication()
        for i, q in enumerate(queries, 1):
            try:
                result = app.process_query(q)
            except Exception as e:
                result = {
                    'success': False,
                    'error': str(e),
                    'processed_query': q,
                    'query_rewritten': False,
                    'translation_info': {
                        'did_translate': False,
                        'source_language': 'unknown',
                        'retries': 0,
                        'success': False,
                        'translated': q,
                        'error': str(e)
                    }
                }
            print_result(i, q, result, show_translated=args.show_translated)
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == '__main__':
    main()