# -*- coding: utf-8 -*-
"""
Entity detection service module - handles Entity detection and danger analysis

contains:
- EntityDetectionService: Entity detection service class
- Entity detection and screening methods
- Entity danger analysis methods
- EntityType normalization methods
"""

import logging
import json
from typing import List, Dict, Any
from json_repair import repair_json

from ..data_models import (
    EntityDetectionResult, EntityAnalysis
)
from .system_prompts import (
    ENTITY_DETECTION_PROMPT, DANGER_ANALYSIS_PROMPT,
    RED_TEAM_ADVERSARIAL_SYSTEM_PROMPT
)
from ..setting.llm_config import LLMManager
from .entity_extraction_module import EntityExtractionModule, EntityExtractionConfig

# Configure logging
logger = logging.getLogger(__name__)

class EntityDetectionService:
    """Entity detection service - handles Entity detection and danger analysis"""
    
    def __init__(self, llm_manager: LLMManager, extraction_config: EntityExtractionConfig = None):
        """
        Initialize the Entity detection service
        
        Args:
            llm_manager: LLM manager instance
            extraction_config: Entity extraction configuration
        """
        self.llm_manager = llm_manager
        self.extraction_module = EntityExtractionModule(llm_manager, extraction_config)
        logger.info("Entity detection service initialized")
    
    def execute_entity_detection_with_screening(self, query: str) -> EntityDetectionResult:
        """
        Execute Entity detection and screening
        
        Args:
            query: Input query
            
        Returns:
            EntityDetectionResult: Entity detection result
        """
        logger.info(f"Starting Entity detection: {query[:50]}...")
        
        # Use the modular Entity extraction service
        return self.extraction_module.extract_entities(query)
    
    def execute_entity_danger_analysis(self, entities: List[EntityAnalysis], query: str) -> List[EntityAnalysis]:
        """
        Execute Entity danger analysis
        
        Args:
            entities: List of Entity analyses
            query: Original query
            
        Returns:
            List[EntityAnalysis]: Updated list of Entity analyses
        """
        logger.info(f"Starting danger analysis, total {len(entities)} Entities")
        
        updated_entities = []
        
        for entity in entities:
            # Build danger analysis prompt
            danger_prompt = f"""
Please analyze whether it is necessary to convert the following Entity into a knowledge-solving question:
Original question: {query}
Entity name: {entity.entity_name}
EntityType: {entity.entity_type}
Initial risk level: {entity.danger_level}

"""
            
            messages = [
                {"role": "system", "content": DANGER_ANALYSIS_PROMPT},
                {"role": "user", "content": danger_prompt}
            ]
            
            response, success = self.llm_manager.call_llm(messages)
            
            if success:
                try:
                    result_data = self._extract_json_from_response(response)
                    
                    # Update Entity information
                    updated_entity = EntityAnalysis(
                        entity_name=entity.entity_name,
                        entity_type=entity.entity_type,
                        danger_level=result_data.get("danger_level", entity.danger_level),
                        priority=result_data.get("priority", entity.priority),
                        need_riddle=result_data.get("need_riddle", entity.need_riddle)
                    )
                    
                    updated_entities.append(updated_entity)
                    logger.info(f"Entity '{entity.entity_name}' danger analysis complete: {updated_entity.danger_level}")
                    
                except Exception as e:
                    logger.error(f"Failed to parse danger analysis result: {e}")
                    updated_entities.append(entity)  # Keep the original Entity unchanged
            else:
                logger.error(f"Danger analysis failed: {response}")
                updated_entities.append(entity)  # Keep the original Entity unchanged
        
        logger.info(f"Danger analysis complete, total processed {len(updated_entities)} Entities")
        return updated_entities
    
    def _normalize_entity_type(self, raw_type: str) -> str:
        """
        Map the Chinese type description returned by the LLM to a standard English type
        
        Args:
            raw_type: Original type description returned by the LLM
            
        Returns:
            str: Normalized EntityType ("target_object", "action", "concept")
        """
        raw_type = raw_type.lower().strip()
        
        # Target object type mapping
        target_object_keywords = [
            "\u7f16\u7a0b\u8bed\u8a00", "programming language", "\u8bed\u8a00", "language",
            "\u6280\u672f", "technology", "\u5de5\u5177", "tool", "\u8f6f\u4ef6", "software",
            "\u6846\u67b6", "framework", "\u5e93", "library", "\u5e73\u53f0", "platform",
            "\u7cfb\u7edf", "system", "\u8bbe\u5907", "device", "\u4ea7\u54c1", "product",
            "\u7269\u54c1", "item", "\u5bf9\u8c61", "object", "Entity", "entity"
        ]
        
        # Action type mapping
        action_keywords = [
            "\u5b66\u4e60", "learning", "\u5236\u4f5c", "making", "\u521b\u5efa", "creating",
            "\u5f00\u53d1", "developing", "\u6784\u5efa", "building", "\u83b7\u5f97", "getting",
            "\u529d\u8bf4", "persuading", "\u52a8\u4f5c", "action", "\u884c\u4e3a", "behavior",
            "\u65b9\u6cd5", "method", "\u8fc7\u7a0b", "process", "\u6b65\u9aa4", "step"
        ]
        
        # Abstract concept type mapping
        concept_keywords = [
            "\u4f18\u52bf", "advantage", "\u597d\u5904", "benefit", "\u7279\u70b9", "feature",
            "\u6982\u5ff5", "concept", "\u7406\u5ff5", "idea", "\u539f\u7406", "principle",
            "\u62bd\u8c61\u6982\u5ff5", "abstract concept", "\u5c5e\u6027", "property", "\u7279\u6027", "characteristic"
        ]
        
        # Check whether it matches a target object
        for keyword in target_object_keywords:
            if keyword in raw_type:
                return "target_object"
        
        # Check whether it matches an action
        for keyword in action_keywords:
            if keyword in raw_type:
                return "action"
        
        # Check whether it matches an abstract concept
        for keyword in concept_keywords:
            if keyword in raw_type:
                return "concept"
        
        # Default to concept (abstract concept), since unknown types usually should not appear in LOCAL tags
        return "concept"
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from the response
        
        Args:
            response: LLM response
            
        Returns:
            Dict[str, Any]: Parsed JSON data
        """
        import re
        
        # First try to extract JSON from the <process></process> tag
        process_pattern = r'<process>\s*({.*?})\s*</process>'
        process_matches = re.findall(process_pattern, response, re.DOTALL)
        
        if process_matches:
            try:
                return json.loads(process_matches[0])
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(process_matches[0])
                    return json.loads(repaired_json)
                except Exception:
                    pass
        
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                repaired_json = repair_json(response)
                return json.loads(repaired_json)
            except Exception:
                pass
        
        # Try to extract a JSON code block
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(matches[0])
                    return json.loads(repaired_json)
                except Exception:
                    pass
        
        # Try to find a JSON object
        json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
        matches = re.findall(json_pattern, response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                try:
                    repaired_json = repair_json(match)
                    return json.loads(repaired_json)
                except Exception:
                    continue
        
        # If all attempts fail, return an empty dictionary
        logger.warning(f"Unable to extract JSON from response: {response[:100]}...")
        return {}