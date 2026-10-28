#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image generation utility module.
Provides image generation with a simple retry mechanism.

Three backends, selected via ``config.IMAGE_GEN_BACKEND``:

``local_pillow`` (default)
    Deterministic local rendering of an "entity board" via Pillow.
    No GPU, no network, fully reproducible.

``draw_things``
    Local generative image via the Draw Things app's HTTP API.
    Enable the API server inside Draw Things (default http://127.0.0.1:7860).

``remote``
    Any OpenAI-compatible endpoint that answers ``chat.completions`` with a
    markdown image link (``![image](data:image/...;base64,...)``).
"""

import base64
import json
import os
import time
from typing import Any, List, Optional

from openai import OpenAI
import config


def _generate_ollama_image(prompt: str, output_path: str, max_retries: int) -> bool:
    """Generate image using Ollama native API (e.g., x/z-image-turbo)."""
    import requests

    endpoint = "http://localhost:11434/api/generate"
    model = getattr(config, "IMAGE_GEN_MODEL", "x/z-image-turbo:latest")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Calling Ollama native image API ({attempt}/{max_retries}): {endpoint}, model={model}")
            response = requests.post(endpoint, json=payload, timeout=config.IMAGE_GEN_TIMEOUT)
            response.raise_for_status()
            body = response.json()
            # Try common fields
            image_b64 = body.get("image") or body.get("images")
            if isinstance(image_b64, list) and image_b64:
                image_b64 = image_b64[0]
            if not image_b64 and body.get("response"):
                resp = body["response"]
                # Sometimes response contains base64 or markdown
                if isinstance(resp, str) and resp.strip().startswith("data:image"):
                    image_b64 = resp.strip().split(",", 1)[1]
            if not image_b64:
                raise ValueError(f"Ollama returned no image: {str(body)[:200]}")
            if isinstance(image_b64, str) and "," in image_b64 and image_b64.strip().startswith("data:image"):
                image_b64 = image_b64.split(",", 1)[1]
            image_data = base64.b64decode(image_b64)
            if len(image_data) < 100:
                raise ValueError(f"Image data too small: {len(image_data)} bytes")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"[SUCCESS] Image generated and saved: {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Ollama image attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


def _generate_remote(prompt: str, output_path: str, max_retries: int) -> bool:
    """Original OpenAI-compatible image generation path."""
    print(f"[INFO] Initializing image generation client...")
    print(f"[INFO] Using model: {config.IMAGE_GEN_MODEL}")
    print(f"[INFO] API Base URL: {config.IMAGE_GEN_BASE_URL}")
    print(f"[INFO] Maximum retries: {max_retries}")
    if not config.IMAGE_GEN_API_KEY or not config.IMAGE_GEN_BASE_URL:
        print("[WARN] Image generation API is not configured; skipping generation. Set IMAGE_GEN_API_KEY and IMAGE_GEN_BASE_URL.")
        return False
    client = OpenAI(api_key=config.IMAGE_GEN_API_KEY, base_url=config.IMAGE_GEN_BASE_URL)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[INFO] Attempt {attempt}/{max_retries}: calling API to generate image...")

            response = client.chat.completions.create(
                model=config.IMAGE_GEN_MODEL,
                messages=[{"role": "user", "content": f"Generate image: {prompt}"}],
                timeout=config.IMAGE_GEN_TIMEOUT,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("API returned an empty response")

            if content.startswith('![image](') and content.endswith(')'):
                url_start = content.find('(') + 1
                url_end = content.rfind(')')
                image_url = content[url_start:url_end]

                if image_url.startswith('data:image/'):
                    base64_data = image_url.split(',', 1)[1] if ',' in image_url else image_url
                    image_data = base64.b64decode(base64_data)
                    if len(image_data) < 100:
                        raise ValueError(f"Image data is too small and may be invalid: {len(image_data)} bytes")
                    with open(output_path, 'wb') as f:
                        f.write(image_data)
                    print(f"[SUCCESS] Image generated and saved: {output_path}")
                    return True

                elif image_url.startswith('http'):
                    import requests
                    img_response = requests.get(image_url, timeout=120)
                    img_response.raise_for_status()
                    if len(img_response.content) < 100:
                        raise ValueError(f"Image data is too small and may be invalid: {len(img_response.content)} bytes")
                    with open(output_path, 'wb') as f:
                        f.write(img_response.content)
                    print(f"[SUCCESS] Image downloaded and saved: {output_path}")
                    return True
                else:
                    raise ValueError(f"Unrecognized image URL format: {image_url[:50]}...")
            else:
                raise ValueError(f"Unexpected response format; first 100 characters: {content[:100]}")

        except Exception as e:
            print(f"[ERROR] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"[INFO] Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print(f"[ERROR] Maximum retries reached; image generation failed")
                import traceback
                traceback.print_exc()
                return False

    return False


def _generate_draw_things(prompt: str, output_path: str, max_retries: int) -> bool:
    """Local generative image via the Draw Things HTTP API.

    Draw Things returns base64 image data; the API server must be enabled in the
    app (Settings -> Advanced -> API Server), default port 7860.
    """
    import requests

    endpoint = getattr(config, "DRAW_THINGS_ENDPOINT", "http://127.0.0.1:7860/api/v1/generate")
    payload = {
        "prompt": prompt,
        "negative_prompt": getattr(config, "DRAW_THINGS_NEGATIVE_PROMPT", ""),
        "seed": 42,
        "steps": int(getattr(config, "DRAW_THINGS_STEPS", 20)),
        "sampler": getattr(config, "DRAW_THINGS_SAMPLER", "DPM++ 2M Karras"),
        "scale": float(getattr(config, "DRAW_THINGS_CFG_SCALE", 7.0)),
        "width": config.IMAGE_GEN_WIDTH,
        "height": config.IMAGE_GEN_HEIGHT,
        "batch": 1,
    }

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[INFO] Calling Draw Things ({attempt}/{max_retries}): {endpoint}")
            response = requests.post(endpoint, json=payload, timeout=config.IMAGE_GEN_TIMEOUT)
            response.raise_for_status()
            body = response.json()
            image_b64 = body.get("image")
            if not image_b64:
                raise ValueError(f"Draw Things response contains no image: {str(body)[:200]}")
            image_data = base64.b64decode(image_b64)
            if len(image_data) < 100:
                raise ValueError(f"Image data is too small: {len(image_data)} bytes")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(image_data)
            print(f"[SUCCESS] Image generated and saved: {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Draw Things attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


def _generate_local_pillow(prompt: str, output_path: str, max_retries: int,
                           entities: Optional[List[Any]]) -> bool:
    """Deterministic local rendering of the entity board."""
    from tools.local_image_generator import render_entity_board, render_from_prompt

    for attempt in range(1, max_retries + 1):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            if entities:
                ok = render_entity_board(
                    entities, output_path,
                    width=config.IMAGE_GEN_WIDTH, height=config.IMAGE_GEN_HEIGHT,
                )
            else:
                print("[WARN] No entity list provided; falling back to a prompt card image (visual grounding quality will decrease)")
                ok = render_from_prompt(
                    prompt, output_path,
                    width=config.IMAGE_GEN_WIDTH, height=config.IMAGE_GEN_HEIGHT,
                )
            if not ok:
                raise ValueError("Local rendering failed")
            print(f"[SUCCESS] Local entity board generated: {output_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Local rendering attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(1)
    return False


def generate_image_with_retry(prompt: str, output_path: str, max_retries: int = 3,
                              entities: Optional[List[Any]] = None) -> bool:
    """Image generation with retries.

    Args:
        prompt: Image generation prompt
        output_path: Output path
        max_retries: Max retry attempts
        entities: Structured entity list, used by the local Pillow backend to
            render one panel per entity.

    Returns:
        bool: Whether generation succeeded
    """
    backend = getattr(config, "IMAGE_GEN_BACKEND", "local_pillow").lower()

    if backend == "local_pillow":
        return _generate_local_pillow(prompt, output_path, max_retries, entities)
    if backend == "draw_things":
        return _generate_draw_things(prompt, output_path, max_retries)
    if backend in ("ollama_image", "ollama_zimage", "zimage"):
        return _generate_ollama_image(prompt, output_path, max_retries)
    return _generate_remote(prompt, output_path, max_retries)
