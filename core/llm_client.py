# -*- coding: utf-8 -*-
"""LLM client - unified API invocation interface."""

import json
import time
import base64
from typing import Dict, List, Any, Optional, Union
from openai import OpenAI
import config


def detect_image_mime(image_data: str) -> str:
    """Detect the MIME type of a base64-encoded image from its magic bytes.

    The original implementation hardcoded ``image/png``, which silently breaks
    for the JPEG payloads produced by ``tools/image_composer.py``.
    """
    if not image_data:
        return "image/png"
    try:
        header = base64.b64decode(image_data[:32], validate=False)
    except Exception:
        header = b""
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith(b"GIF8"):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    return "image/png"


class LLMClient:
    """Unified LLM API client - supports separate configuration for text and vision models."""
    
    def __init__(self):
        if config.TEXT_API_KEY and config.TEXT_BASE_URL:
            self.text_client = OpenAI(
                api_key=config.TEXT_API_KEY,
                base_url=config.TEXT_BASE_URL
            )
        else:
            self.text_client = None
        if config.VISION_API_KEY and config.VISION_BASE_URL:
            self.vision_client = OpenAI(
                api_key=config.VISION_API_KEY,
                base_url=config.VISION_BASE_URL
            )
        else:
            self.vision_client = None
        self.client = self.text_client
    
    def call_text_model(self, prompt: str, task_type: str = "entity_extraction", 
                       **kwargs) -> str:
        """Call the text model."""
        task_config = config.LLM_TASKS.get(task_type, config.LLM_TASKS["entity_extraction"])
        
        # Merge configuration parameters
        params = {
            "model": task_config["model"],
            "temperature": task_config["temperature"],
            "max_tokens": task_config["max_tokens"],
            **kwargs
        }
        
        retry_limit = task_config["retry_limit"]
        
        if self.text_client is None:
            raise RuntimeError("TEXT client not configured. Set TEXT_API_KEY and TEXT_BASE_URL.")
        
        for attempt in range(retry_limit):
            try:
                response = self.text_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                        {"role": "user", "content": prompt}
                    ],
                    **params
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if attempt == retry_limit - 1:
                    raise RuntimeError(f"LLM API call failed after {retry_limit} attempts: {e}")
                time.sleep(1)  # Simple retry delay
    
    def call_vision_model(self, prompt: str, image_data: str, 
                         task_type: str = "vlm_recognition", **kwargs) -> str:
        """Call the vision model."""
        task_config = config.LLM_TASKS.get(task_type, config.LLM_TASKS["vlm_recognition"])
        
        params = {
            "model": task_config["model"],
            "temperature": task_config["temperature"],
            "max_tokens": task_config["max_tokens"],
            **kwargs
        }
        
        retry_limit = task_config["retry_limit"]
        
        if self.vision_client is None:
            raise RuntimeError("VISION client not configured. Set VISION_API_KEY and VISION_BASE_URL.")
        
        mime_type = detect_image_mime(image_data)
        
        for attempt in range(retry_limit):
            try:
                response = self.vision_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes images accurately."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                        ]}
                    ],
                    **params
                )
                return response.choices[0].message.content
            
            except Exception as e:
                if attempt == retry_limit - 1:
                    raise RuntimeError(f"Vision API call failed after {retry_limit} attempts: {e}")
                time.sleep(1)


# Global client instance
llm_client = LLMClient()
