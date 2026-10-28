# -*- coding: utf-8 -*-
"""VLM recognition module with question-driven prompts."""

import json
import base64
from typing import Dict, List, Any, Optional
from process_utils import extract_process_blocks, load_block_json, validate_schema
from .llm_client import llm_client
import prompts
import config


class VLMRecognizer:
    """VLM question-driven recognizer that analyzes images and identifies entities using a multimodal LLM."""
    
    def recognize_image(self, image_path: str) -> Dict[str, Any]:
        """Perform question-driven recognition on the image."""
        
        # Read and encode the image
        image_data = self._encode_image(image_path)
        
        # Get the recognition prompt
        prompt = prompts.get_prompt("vlm_recognition")
        response = llm_client.call_vision_model(prompt, image_data, "vlm_recognition")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        recognition_blocks = [b for b in blocks if b.name == "Recognition"]
        
        if not recognition_blocks:
            raise ValueError("No Recognition found in VLM response")
        
        recognition_data = load_block_json(recognition_blocks[0])
        
        # Validate basic structure
        if "identified_elements" not in recognition_data:
            raise ValueError("Recognition result missing 'identified_elements'")
        if "overall_understanding" not in recognition_data:
            raise ValueError("Recognition result missing 'overall_understanding'")
        
        return recognition_data
    
    def recognize_from_base64(self, image_base64: str) -> Dict[str, Any]:
        """Recognize from base64 image data."""
        
        prompt = prompts.get_prompt("vlm_recognition")
        response = llm_client.call_vision_model(prompt, image_base64, "vlm_recognition")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        recognition_blocks = [b for b in blocks if b.name == "Recognition"]
        
        if not recognition_blocks:
            raise ValueError("No Recognition found in VLM response")
        
        recognition_data = load_block_json(recognition_blocks[0])
        
        # Validate basic structure
        if "identified_elements" not in recognition_data:
            raise ValueError("Recognition result missing 'identified_elements'")
        if "overall_understanding" not in recognition_data:
            raise ValueError("Recognition result missing 'overall_understanding'")
        
        return recognition_data
    
    def evaluate_recognition_quality(self, original_entities: List[Dict[str, Any]], 
                                   recognition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate recognition quality."""
        
        # Prepare assessment data
        entities_json = json.dumps(original_entities, ensure_ascii=False, indent=2)
        recognition_json = json.dumps(recognition_result, ensure_ascii=False, indent=2)
        
        prompt = prompts.get_prompt("recognition_quality", 
                                  original_entities=entities_json,
                                  recognition_result=recognition_json)
        
        response = llm_client.call_text_model(prompt, "quality_assessment")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        quality_blocks = [b for b in blocks if b.name == "Quality"]
        
        if not quality_blocks:
            raise ValueError("No Quality assessment found in LLM response")
        
        quality_data = load_block_json(quality_blocks[0])
        
        return quality_data
    
    def filter_recognized_entities(self, original_entities: List[Dict[str, Any]], 
                                 recognition_result: Dict[str, Any],
                                 confidence_threshold: float = None) -> List[Dict[str, Any]]:
        """Filter entities based on recognition results, keeping only recognized entities (using LLM semantic matching)."""
        
        if confidence_threshold is None:
            confidence_threshold = config.QUALITY_THRESHOLDS["recognition_confidence"]
        
        identified_elements = recognition_result.get("identified_elements", [])
        
        # Use the LLM for semantic matching
        entities_json = json.dumps(original_entities, ensure_ascii=False, indent=2)
        elements_json = json.dumps(identified_elements, ensure_ascii=False, indent=2)
        
        matching_prompt = f"""Semantically match the original entities to the identified image elements.

Original entity list:
{entities_json}

Identified image elements:
{elements_json}

Requirements:
1. Determine whether each original entity has a corresponding element in the recognition results
2. Match based on semantic similarity, not exact text matching
3. Only elements with confidence >= {confidence_threshold} should be considered matches
4. Return successfully matched entities and their corresponding recognition information

Output format: <process name="EntityMatching">
{{
  "matched_entities": [
    {{
      "entity_name": "original entity name",
      "entity_canonical": "canonical name",
      "matched_element": "matched element description",
      "confidence": 0.9,
      "evidence": "matching evidence",
      "location": "location"
    }}
  ],
  "unmatched_entities": ["unmatched entity names"]
}}
</process>"""

        response = llm_client.call_text_model(matching_prompt, "entity_matching")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        matching_blocks = [b for b in blocks if b.name == "EntityMatching"]
        
        if not matching_blocks:
            raise ValueError("No EntityMatching found in LLM response")
        
        matching_data = load_block_json(matching_blocks[0])
        
        # Build the return result
        filtered_entities = []
        for match in matching_data.get("matched_entities", []):
            # Find the corresponding original entity
            for entity in original_entities:
                if entity["name"] == match["entity_name"] or entity["canonical"] == match["entity_canonical"]:
                    filtered_entities.append({
                        **entity,
                        "recognition_confidence": match.get("confidence", 0),
                        "recognition_evidence": match.get("evidence", ""),
                        "visual_location": match.get("location", "")
                    })
                    break
        
        # Record unmatched entities
        for unmatched in matching_data.get("unmatched_entities", []):
            print(f"Entity '{unmatched}' not recognized in image")
        
        return filtered_entities
    
    def _encode_image(self, image_path: str) -> str:
        """Encode the image as base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")


class RecognitionGate:
    """Recognition gate that decides which entities can be redacted based on recognition results."""
    
    def __init__(self, recognizer: VLMRecognizer):
        self.recognizer = recognizer
    
    def gate_entities(self, original_entities: List[Dict[str, Any]], 
                     image_path: str) -> Dict[str, Any]:
        """Run recognition gating and return redactable entities and recognition results."""
        
        # 1. VLM recognition
        recognition_result = self.recognizer.recognize_image(image_path)
        
        # 2. Quality assessment
        quality_assessment = self.recognizer.evaluate_recognition_quality(
            original_entities, recognition_result
        )
        
        # 3. Filter entities
        recognized_entities = self.recognizer.filter_recognized_entities(
            original_entities, recognition_result
        )
        
        # 4. Decision gating
        gate_decision = self._make_gate_decision(quality_assessment, recognized_entities)
        
        return {
            "recognition_result": recognition_result,
            "quality_assessment": quality_assessment,
            "recognized_entities": recognized_entities,
            "gate_decision": gate_decision,
            "can_proceed": gate_decision["allow_redaction"]
        }
    
    def _make_gate_decision(self, quality_assessment: Dict[str, Any], 
                          recognized_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make a gating decision based on quality assessment and recognition results."""
        
        hit_rate = quality_assessment.get("hit_rate", 0)
        confidence_score = quality_assessment.get("confidence_score", 0)
        num_recognized = len(recognized_entities)
        
        # Decision logic
        allow_redaction = (
            hit_rate >= config.QUALITY_THRESHOLDS["recognition_confidence"] and
            confidence_score >= config.QUALITY_THRESHOLDS["recognition_confidence"] and
            num_recognized >= 1  # At least one entity recognized
        )
        
        return {
            "allow_redaction": allow_redaction,
            "hit_rate": hit_rate,
            "confidence_score": confidence_score,
            "num_recognized": num_recognized,
            "decision_reason": self._get_decision_reason(allow_redaction, hit_rate, confidence_score, num_recognized)
        }
    
    def _get_decision_reason(self, allow: bool, hit_rate: float, confidence: float, num_recognized: int) -> str:
        """Get the decision reason."""
        if allow:
            return f"Recognition quality is good: hit rate {hit_rate:.2f}, confidence {confidence:.2f}, {num_recognized} entities recognized"
        else:
            reasons = []
            if hit_rate < config.QUALITY_THRESHOLDS["recognition_confidence"]:
                reasons.append(f"hit rate too low ({hit_rate:.2f})")
            if confidence < config.QUALITY_THRESHOLDS["recognition_confidence"]:
                reasons.append(f"confidence too low ({confidence:.2f})")
            if num_recognized < 1:
                reasons.append("no entities recognized")
            return f"Recognition quality is insufficient: {', '.join(reasons)}"


# Global instance
vlm_recognizer = VLMRecognizer()
recognition_gate = RecognitionGate(vlm_recognizer)
