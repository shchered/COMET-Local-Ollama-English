# -*- coding: utf-8 -*-
"""
Entity extraction module - standalone Entity extraction LLM call service

This module modularizes the LLM call process for Entity extraction, making it easy to swap models
contains:
- EntityExtractionModule: Entity extraction module class
- Entity extraction methods supporting multiple models
- Unified interface design
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from json_repair import repair_json

from ..data_models import EntityAnalysis, EntityDetectionResult
from .system_prompts import get_entity_detection_prompt
from ..setting.llm_config import LLMManager, RETRY_CONFIG, ENTITY_EXTRACTION_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EntityExtractionConfig:
    """Entity extraction configuration"""
    def __post_init__(self):
        # Load parameters from the unified configuration
        config = ENTITY_EXTRACTION_CONFIG
        self.model_name = config['model_name']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.timeout = config['timeout']
        self.retry_delay = config['retry_delay']
        self.enable_danger_analysis = config['enable_danger_analysis']
        self.fallback_models = config['fallback_models']

class EntityExtractionModule:
    """Entity extraction module - standalone LLM call service"""
    
    def __init__(self, llm_manager: LLMManager, config: Optional[EntityExtractionConfig] = None):
        """
        Initialize the Entity extraction module
        
        Args:
            llm_manager: LLM manager instance
            config: Entity extraction configuration
        """
        self.llm_manager = llm_manager
        self.config = config or EntityExtractionConfig()
        logger.info(f"Entity extraction module initialized, using model: {self.config.model_name}")
    
    def extract_entities(self, query: str) -> EntityDetectionResult:
        """
        Extract Entities from the query
        
        Args:
            query: Input query
            
        Returns:
            EntityDetectionResult: Entity detection result
        """
        logger.info(f"Starting Entity extraction: {query[:50]}...")
        
        # Build the Entity detection prompt
        detection_prompt = f"""
Now please analyze the following question:
Question: {query}

"""
        
        messages = [
            {"role": "system", "content": get_entity_detection_prompt()},
            {"role": "user", "content": detection_prompt}
        ]
        
        # Add retry mechanism
        import time
        max_retries = RETRY_CONFIG['entity_extraction_retries']
        for attempt in range(max_retries):
            try:
                response, success = self.llm_manager.call_llm(messages)
                
                if success:
                    try:
                        # Parse the JSON response
                        result_data = self._extract_json_from_response(response)
                        
                        # Build the EntityAnalysis list
                        entities = []
                        for entity_data in result_data.get("entities", []):
                            # Map the Chinese type description returned by the LLM to a standard English type
                            raw_type = entity_data.get("type", "").lower()
                            entity_type = self._normalize_entity_type(raw_type)
                            
                            entities.append(EntityAnalysis(
                                entity_name=entity_data.get("name", ""),
                                entity_type=entity_type,
                                danger_level=entity_data.get("risk_level", "safe"),
                                priority=1,
                                need_riddle=entity_data.get("risk_level", "safe") in ["mild", "dangerous"]
                            ))
                        
                        result = EntityDetectionResult(
                            detected_entities=entities,
                            processing_complete=True
                        )
                        
                        logger.info(f"Entity detection complete, found {len(entities)} Entities")
                        return result
                        
                    except Exception as e:
                        if attempt == max_retries - 1:
                            return EntityDetectionResult(
                                detected_entities=[],
                                processing_complete=False
                            )
                else:
                    if attempt == max_retries - 1:
                        return EntityDetectionResult(
                            detected_entities=[],
                            processing_complete=False
                        )
                
                # Wait then retry
                if attempt < max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    return EntityDetectionResult(
                        detected_entities=[],
                        processing_complete=False
                    )
                # Wait then retry
                if attempt < max_retries - 1:
                    time.sleep(self.config.retry_delay)
    
    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON data from the response
        
        Args:
            response: LLM response
            
        Returns:
            Optional[Dict[str, Any]]: Parsed JSON data
        """
        try:
            # Try to parse JSON directly
            if response.strip().startswith('{'):
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    try:
                        # Use json_repair to repair corrupted JSON
                        repaired_json = repair_json(response)
                        return json.loads(repaired_json)
                    except Exception:
                        pass
            
            # Find JSON in the <process> tag
            process_match = re.search(r'<process>\s*({.*?})\s*</process>', response, re.DOTALL)
            if process_match:
                try:
                    return json.loads(process_match.group(1))
                except json.JSONDecodeError:
                    try:
                        # Use json_repair to repair corrupted JSON
                        repaired_json = repair_json(process_match.group(1))
                        return json.loads(repaired_json)
                    except Exception:
                        pass
            
            # Find any JSON structure
            json_match = re.search(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    try:
                        # Use json_repair to repair corrupted JSON
                        repaired_json = repair_json(json_match.group(0))
                        return json.loads(repaired_json)
                    except Exception:
                        pass
            
            logger.error(f"Unable to extract JSON from response: {response[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"JSON parsing error: {e}")
            return None
    
    def _normalize_entity_type(self, raw_type: str) -> str:
        """
        Normalize EntityType - directly return the type output by the LLM
        
        Args:
            raw_type: Original type description
            
        Returns:
            str: Normalized EntityType
        """
        # Directly return the EntityType output by the LLM, no dictionary matching needed
        return raw_type.strip()
    
    def set_model(self, model_name: str):
        """
        Set the model to use
        
        Args:
            model_name: Model name
        """
        self.config.model_name = model_name
        logger.info(f"Entity extraction module switched to model: {model_name}")
    
    def get_current_model(self) -> str:
        """
        Get the currently used model
        
        Returns:
            str: Current model name
        """
        return self.config.model_name
    
    def get_available_models(self) -> List[str]:
        """
        Get the list of available models
        
        Returns:
            List[str]: List of available models
        """
        return [self.config.model_name] + self.config.fallback_models