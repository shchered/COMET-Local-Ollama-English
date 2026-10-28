# -*- coding: utf-8 -*-
"""Text redaction and adversarial rewriting module."""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from process_utils import extract_process_blocks, load_block_json, validate_schema
from .llm_client import llm_client
import prompts
import config


class TextRedactor:
    """Redacts text based on recognition results and gating."""
    
    def redact_text(self, original_text: str, recognition_result: Dict[str, Any], 
                   recognized_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Redact text guided by recognition output."""
        
        # Prepare data
        recognition_json = json.dumps(recognition_result, ensure_ascii=False, indent=2)
        
        prompt = prompts.get_prompt("text_redaction", 
                                  recognition_json=recognition_json,
                                  original_text=original_text)
        
        response = llm_client.call_text_model(prompt, "text_redaction")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        redaction_blocks = [b for b in blocks if b.name == "Redaction"]
        
        if not redaction_blocks:
            raise ValueError("No Redaction found in LLM response")
        
        redaction_data = load_block_json(redaction_blocks[0])
        
        # Validate basic structure
        if "mappings" not in redaction_data:
            raise ValueError("Redaction result missing 'mappings'")
        if "redacted_text" not in redaction_data:
            raise ValueError("Redaction result missing 'redacted_text'")
        
        # Add metadata
        redaction_data["original_text"] = original_text
        redaction_data["num_redactions"] = len(redaction_data["mappings"])
        
        return redaction_data
    
class AdversarialRewriter:
    """Adversarial rewriter that obfuscates and aligns redacted text."""
    
    def rewrite_text(self, redaction_result: Dict[str, Any], 
                    recognition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform adversarial rewriting on the redacted text."""
        
        redacted_text = redaction_result.get("redacted_text", "")
        image_understanding = recognition_result.get("overall_understanding", "")
        
        prompt = prompts.get_prompt("adversarial_rewrite",
                                  redacted_text=redacted_text,
                                  image_understanding=image_understanding)
        
        response = llm_client.call_text_model(prompt, "adversarial_rewrite")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        rewrite_blocks = [b for b in blocks if b.name == "Rewrite"]
        
        if not rewrite_blocks:
            raise ValueError("No Rewrite found in LLM response")
        
        rewrite_data = load_block_json(rewrite_blocks[0])
        
        # Validate basic structure
        if "surface_form" not in rewrite_data:
            raise ValueError("Rewrite result missing 'surface_form'")
        if "coref_style" not in rewrite_data:
            raise ValueError("Rewrite result missing 'coref_style'")
        if "image_references" not in rewrite_data:
            raise ValueError("Rewrite result missing 'image_references'")
        
        # Add metadata
        rewrite_data["original_redacted"] = redacted_text
        rewrite_data["rewrite_quality"] = self._assess_rewrite_quality(rewrite_data)
        
        return rewrite_data
    
    def _assess_rewrite_quality(self, rewrite_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess rewrite quality using an LLM."""
        
        surface_form = rewrite_data.get("surface_form", "")
        coref_style = rewrite_data.get("coref_style", "")
        image_refs = rewrite_data.get("image_references", [])
        
        # Build the assessment prompt
        assessment_prompt = f"""Evaluate the quality of the following rewritten text (0-1 scoring):

Rewritten text: {surface_form}

Reference style: {coref_style}

Image references: {image_refs}

Evaluation dimensions:
1. image_integration: naturalness and effectiveness of image references
2. pronoun_usage: quality of pronoun and reference usage
3. fluency: language fluency and readability
4. obfuscation: degree of obfuscation (information-hiding effect)

Output format: <process name="QualityScore">
{{
  "image_integration": 0.9,
  "pronoun_usage": 0.85,
  "fluency": 0.9,
  "obfuscation": 0.8,
  "overall_score": 0.86
}}
</process>"""

        response = llm_client.call_text_model(assessment_prompt, "quality_assessment")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        quality_blocks = [b for b in blocks if b.name == "QualityScore"]
        
        if not quality_blocks:
            raise ValueError("No QualityScore found in LLM response")
        
        quality_scores = load_block_json(quality_blocks[0])
        
        return quality_scores
    
class TextProcessor:
    """Text processor that integrates redaction and rewriting."""
    
    def __init__(self):
        self.redactor = TextRedactor()
        self.rewriter = AdversarialRewriter()
    
    def process_text(self, original_text: str, recognition_result: Dict[str, Any],
                    recognized_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Complete text processing pipeline: redaction → rewriting."""
        
        print("📝 Starting text redaction...")
        
        # 1. Text redaction
        redaction_result = self.redactor.redact_text(
            original_text, recognition_result, recognized_entities
        )
        
        print(f"✅ Redaction complete, processed {redaction_result['num_redactions']} entities")
        
        # 2. Adversarial rewriting
        print("🔄 Starting adversarial rewriting...")
        
        rewrite_result = self.rewriter.rewrite_text(
            redaction_result, recognition_result
        )
        
        print(f"✅ Rewriting complete, quality score: {rewrite_result['rewrite_quality']['overall_score']:.2f}")
        
        # 3. Integrate results
        return {
            "redaction": redaction_result,
            "rewrite": rewrite_result,
            "final_text": rewrite_result["surface_form"],
            "processing_metadata": {
                "num_redactions": redaction_result["num_redactions"],
                "rewrite_quality": rewrite_result["rewrite_quality"],
                "image_references": rewrite_result["image_references"]
            }
        }
    
    def evaluate_processing_quality(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall processing quality."""
        
        final_text = processing_result.get("final_text", "")
        
        prompt = prompts.get_prompt("readability_assessment", rewritten_text=final_text)
        response = llm_client.call_text_model(prompt, "quality_assessment")
        
        # Parse the <process> block
        blocks = extract_process_blocks(response)
        readability_blocks = [b for b in blocks if b.name == "Readability"]
        
        if not readability_blocks:
            raise ValueError("No Readability assessment found in LLM response")
        
        readability_data = load_block_json(readability_blocks[0])
        
        return readability_data


# Global instance
text_processor = TextProcessor()
