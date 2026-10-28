# -*- coding: utf-8 -*-
"""
Template construction service module - handles global template construction and output formatting

contains:
- TemplateBuilderService: Template construction service class
- Global template construction methods
- Output formatting methods
- Template optimization methods
"""
import re
import logging
import json
from typing import List, Dict, Any, Optional
from json_repair import repair_json

from ..data_models import (
    EntityAnalysis, RiddleResult, FormattedOutput, QualityAssessment, EntityQuestionMapping
)
from .system_prompts import (
    INTELLIGENT_GLOBAL_QUESTION_PROMPT, TEMPLATE_OPTIMIZATION_PROMPT,
    RED_TEAM_ADVERSARIAL_SYSTEM_PROMPT, ADVERSARIAL_AUDIT_PROMPT
)
from ..setting.llm_config import LLMManager, RETRY_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class InformationStructuringService:
    """Information structuring service - handles global template construction and output formatting"""
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the template construction service
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        logger.info("Template construction service initialized")
    
    def execute_global_template_construction(self, original_query: str, riddles: List[RiddleResult], contextual_info: Dict[str, Any] = None) -> FormattedOutput:
        """
        Execute global template construction
        
        Args:
            original_query: Original query
            riddles: List of knowledge-solving question results
            contextual_info: Spatiotemporal background info (scene, time, target_group)
        
        Returns:
            FormattedOutput: Formatted output result
        """
        logger.info(f"Starting global template construction, original query: {original_query[:50]}...")
        logger.info(f"[DEBUG execute_global_template_construction] received {len(riddles)} riddles")
        for idx, riddle in enumerate(riddles):
            logger.info(f"[DEBUG execute_global_template_construction] riddle[{idx}]: entity={riddle.entity_name}, question={riddle.riddle_question[:100]}...")
        
        if contextual_info:
            logger.info(f"[DEBUG] Spatiotemporal background info: {contextual_info}")
        
        try:
            # Build the global template
            global_question = self._construct_global_template(original_query, riddles, contextual_info)
            
            # Extract definition questions
            from .knowledge_identification_service import KnowledgeIdentificationService
            riddle_service = KnowledgeIdentificationService(self.llm_manager)
            definition_mapping = riddle_service.extract_definition_questions_from_global(
                global_question, riddles, original_query
            )
            
            # Build the formatted output
            formatted_output = self._build_formatted_output(
                global_question, definition_mapping, riddles, original_query
            )
            
            logger.info("Global template construction complete")
            return formatted_output
            
        except Exception as e:
            logger.error(f"Global template construction failed: {e}")
            # Return a basic formatted output
            return self._create_fallback_output(original_query, riddles)
    
    def _construct_global_template(self, original_query: str, riddles: List[RiddleResult], contextual_info: Dict[str, Any] = None) -> str:
        """
        Build the global template, including adversarial audit and retry mechanism
        
        Args:
            original_query: Original query
            riddles: List of knowledge-solving question results
            contextual_info: Spatiotemporal background info (scene, time, target_group)
            
        Returns:
            str: Global question template
            
        Raises:
            RuntimeError: When all retries fail
        """
        logger.info("Build global question template")
        
        max_retries = RETRY_CONFIG['adversarial_audit_retries']
        failure_info = ""
        best_question = None
        best_score = -1
        
        for attempt in range(max_retries):
            logger.info(f"Global question generation attempt {attempt + 1}/{max_retries}")
            
            # Generate the global question
            global_question = self._generate_global_question(original_query, riddles, failure_info, contextual_info)
            if not global_question:
                failure_info = "Global question generation failed, please regenerate"
                continue
                
            # Adversarial audit
            audit_result = self._adversarial_audit(global_question, original_query)
            
            # Track the best result
            current_score = 0
            if audit_result["result"] == "pass":
                current_score = 3
            elif audit_result["result"] == "warning":
                current_score = 2
            else:
                current_score = 1
                
            if current_score >= 1:
                best_question = global_question
                best_score = current_score
            
            if audit_result["result"] == "pass":
                # logger.info(f"Global question passed audit: {global_question[:100]}...")
                logger.info(f"Global question passed audit")
                return global_question
            elif audit_result["result"] == "warning":
                logger.warning(f"Global question has minor issues but is acceptable: {audit_result['issues_found']}")
                # return global_question
            else:  # fail
                failure_info = f"Audit failed: {audit_result['issues_found']}"
                logger.warning(f"Global question audit failed, retrying: {failure_info}")
                
        # All retries failed, use the best result or raise an exception
        if best_question:
            logger.warning(f"Global question generation failed multiple times, using best result: {best_question[:100]}...")
            return best_question
        else:
            error_msg = f"Global question generation completely failed, last failure info: {failure_info}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _generate_global_question(self, original_query: str, riddles: List[RiddleResult], failure_info: str = "", contextual_info: Dict[str, Any] = None) -> str:
        """
        Generate the global question
        
        Args:
            original_query: Original query
            riddles: List of knowledge-solving question results
            failure_info: Failure info (used to improve generation)
            contextual_info: Spatiotemporal background info (scene, time, target_group)
            
        Returns:
            str: Generated global question
        """
        logger.info("Generate global question")
        
        if contextual_info is None:
            contextual_info = {}
        
        # Prepare the paired format of Entity list and local questions
        entity_question_pairs = []
        for riddle in riddles:
            entity_question_pairs.append(f"Entity: {riddle.entity_name} -> Solving question: {riddle.riddle_question}")
        
        # Join the paired info with newlines to form a clear format
        local_section = "\n".join(entity_question_pairs)
        
        # Also prepare a simple Entity list for backward compatibility
        entity_list = ", ".join([riddle.entity_name for riddle in riddles])
        
        # Format spatiotemporal background info
        contextual_info_str = "None" 
        if contextual_info:
            parts = []
            # Add question type info (if present)
            question_type = contextual_info.get("question_type", "Unknown")
            parts.append(f"Question type: {question_type}")
            
            if contextual_info.get("scene"):
                parts.append(f"Scene: {contextual_info['scene']}")
            if contextual_info.get("time"):
                parts.append(f"Time: {contextual_info['time']}")
            if contextual_info.get("target_group"):
                parts.append(f"Target group: {contextual_info['target_group']}")
            if parts:
                contextual_info_str = "; ".join(parts)
        
        # Build the global question prompt
        template_prompt = INTELLIGENT_GLOBAL_QUESTION_PROMPT.format(
            original_query=original_query,
            # entity_list=entity_list,
            local_section=local_section,
            contextual_info=contextual_info_str
        )
        
        # If there is failure info, add it to the prompt
        if failure_info:
            template_prompt += f"\n**Improvement requirement (failure issues from previous improvement rounds)**: {failure_info}"
        
        messages = [
            {"role": "system", "content": RED_TEAM_ADVERSARIAL_SYSTEM_PROMPT},
            {"role": "user", "content": template_prompt}
        ]
        
        response, success = self.llm_manager.call_llm(messages)
        
        if success:
            # Extract and parse the JSON response
            try:
                json_data = self._extract_json_from_response(response)
                if json_data and 'global_question' in json_data:
                    global_question = json_data['global_question']
                    logger.info(f"Global question generated successfully: {global_question[:100]}...")
                    return global_question
                else:
                    # If JSON parsing fails, try to extract the process content directly
                    logger.warning(f"JSON parsing failed or global_question field is missing, trying to extract process content")
                    logger.debug(f"Response content: {response[:500]}...")
                    global_question = self._extract_process_content(response)
                    if not global_question or global_question == response.strip():
                        # If extracting process also fails, try to extract any text from the JSON
                        if json_data:
                            # Try to get the first value
                            for key, value in json_data.items():
                                if isinstance(value, str) and len(value) > 10:
                                    global_question = value
                                    logger.warning(f"Extracted content from JSON field '{key}' as the global question")
                                    break
                        if not global_question:
                            global_question = response.strip()
                    logger.warning(f"Using fallback content: {global_question[:100]}...")
                    return global_question
            except Exception as e:
                logger.error(f"Error parsing global question response: {e}")
                logger.debug(f"Original response: {response[:500]}...")
                # Fall back to the original response
                global_question = response.strip()
                return global_question
        else:
            logger.error(f"Global question generation failed: {response}")
            return ""
    
    def _build_formatted_output(self, global_question: str, definition_mapping: Dict[str, str], riddles: List[RiddleResult], original_query: str) -> FormattedOutput:
        """
        Build the formatted output object
        
        Args:
            global_question: Global question
            definition_mapping: Definition question mapping
            riddles: List of knowledge-solving question results
            original_query: Original question
            
        Returns:
            FormattedOutput: Formatted output object
        """
        logger.info("Build formatted output object")
        logger.info(f"[DEBUG] number of riddles: {len(riddles)}")
        logger.info(f"[DEBUG] definition_mapping content: {definition_mapping}")
        logger.info(f"[DEBUG] global_question: {global_question}")
        
        # Build the local question list
        local_questions = []
        local_entities = []
        
        for riddle in riddles:
            if riddle.entity_name in definition_mapping:
                local_questions.append(definition_mapping[riddle.entity_name])
                local_entities.append(riddle.entity_name)
            else:
                # Use the default knowledge-solving question
                local_questions.append(riddle.riddle_question)
                local_entities.append(riddle.entity_name)
        
        # Create the Entity question mapping
        entity_question_mappings = []
        for i, riddle in enumerate(riddles):
            if i < len(local_questions):
                mapping = EntityQuestionMapping(
                    entity_name=riddle.entity_name,
                    entity_type=riddle.entity_type,
                    knowledge_seeking=local_questions[i],
                    quality_score=riddle.quality_score
                )
                entity_question_mappings.append(mapping)
        
        # Build the XML-format template string
        template_parts = [f"<ORIGIN>{original_query}</ORIGIN>"]
        template_parts.append("<GLOBAL>")
        for i, riddle in enumerate(riddles, 1):
            knowledge_question = definition_mapping.get(riddle.entity_name, riddle.riddle_question)
            logger.info(f"[DEBUG] Adding LOCAL_{i}: entity={riddle.entity_name}, question={knowledge_question[:100]}...")
            template_parts.append(f"<LOCAL_{i}>{knowledge_question}</LOCAL_{i}>")
            template_parts.append(f"<ANSWER_{i}>{riddle.entity_name}</ANSWER_{i}>")
        template_parts.append(global_question)
        template_parts.append("</GLOBAL>")
        formatted_template = "".join(template_parts)
        
        logger.info(f"[DEBUG] formatted_template length: {len(formatted_template)}")
        logger.info(f"[DEBUG] first 500 characters of formatted_template: {formatted_template[:500]}")
        
        # Use the original global_question directly as clean_global_question
        # Do not extract from formatted_template, to avoid duplication caused by LOCAL question content
        clean_global_question = global_question
        
        # Create the FormattedOutput object
        formatted_output = FormattedOutput(
            global_question=clean_global_question,
            local_questions=local_questions,
            local_entities=local_entities,
            entity_question_mappings=entity_question_mappings,
            formatted_output=formatted_template
        )
        
        logger.info(f"Formatted output construction complete, contains {len(local_questions)} local questions")
        return formatted_output
    
    def _create_fallback_output(self, original_query: str, riddles: List[RiddleResult]) -> FormattedOutput:
        """
        Create a fallback formatted output
        
        Args:
            original_query: Original query
            riddles: List of knowledge-solving question results
            
        Returns:
            FormattedOutput: Fallback formatted output
        """
        logger.info("Create fallback formatted output")
        
        local_questions = [riddle.riddle_question for riddle in riddles]
        local_entities = [riddle.entity_name for riddle in riddles]
        
        # Create the Entity question mapping
        entity_question_mappings = []
        for i, riddle in enumerate(riddles):
            if i < len(local_questions):
                mapping = EntityQuestionMapping(
                    entity_name=riddle.entity_name,
                    entity_type=riddle.entity_type,
                    knowledge_seeking=local_questions[i],
                    quality_score=riddle.quality_score
                )
                entity_question_mappings.append(mapping)
        
        # Build the correct XML-format template
        template_parts = ["<GLOBAL><ORIGIN>", original_query, "</ORIGIN>"]
        
        for i, riddle in enumerate(riddles, 1):
            template_parts.append(f"<LOCAL_{i}>{riddle.riddle_question}</LOCAL_{i}>")
            template_parts.append(f"<ANSWER_{i}>{riddle.entity_name}</ANSWER_{i}>")
        
        # Use the original query as the global question (fallback case)
        template_parts.append(original_query)
        template_parts.append("</GLOBAL>")
        
        formatted_template = "".join(template_parts)
        
        logger.info(f"Fallback formatted output construction complete, contains {len(local_questions)} local questions")
        
        return FormattedOutput(
            global_question=original_query,
            local_questions=local_questions,
            local_entities=local_entities,
            entity_question_mappings=entity_question_mappings,
            formatted_output=formatted_template
        )
    
    def construct_global_template(self, riddles, original_query: str) -> FormattedOutput:
        """
        Build the global template (compatible with the old interface)
        
        Args:
            riddles: List of knowledge-solving question results
            original_query: Original query
            
        Returns:
            FormattedOutput: Formatted output result
        """
        return self.execute_global_template_construction(original_query, riddles)
    
    def optimize_template_quality(self, formatted_output: FormattedOutput, original_query: str) -> FormattedOutput:
        """
        Optimize template quality
        
        Args:
            formatted_output: Original formatted output
            original_query: Original query
            
        Returns:
            FormattedOutput: Optimized formatted output
        """
        logger.info("Starting template quality optimization")
        
        try:
            # Build the optimization prompt
            optimization_prompt = f"""
{TEMPLATE_OPTIMIZATION_PROMPT}

Original question: {original_query}
Current global question: {formatted_output.global_question}

Current local questions:
{chr(10).join([f"{i+1}. {q}" for i, q in enumerate(formatted_output.local_questions)])}

Corresponding Entities:
{chr(10).join([f"{i+1}. {e}" for i, e in enumerate(formatted_output.local_entities)])}

Please return the optimization result in the following JSON format:
{{
    "optimized_global_question": "Optimized global question",
    "optimized_local_questions": ["Optimized local question 1", "Optimized local question 2"],
    "optimization_notes": "Optimization notes",
    "quality_improvement": {{
        "clarity_improvement": 0.0-1.0,
        "coherence_improvement": 0.0-1.0,
        "completeness_improvement": 0.0-1.0
    }}
}}
"""
            
            messages = [
                {"role": "system", "content": RED_TEAM_ADVERSARIAL_SYSTEM_PROMPT},
                {"role": "user", "content": optimization_prompt}
            ]
            
            response, success = self.llm_manager.call_llm(messages)
            
            if success:
                result_data = self._extract_json_from_response(response)
                
                # Update the formatted output
                optimized_global = result_data.get("optimized_global_question", formatted_output.global_question)
                optimized_locals = result_data.get("optimized_local_questions", formatted_output.local_questions)
                
                # Ensure the number of local questions matches
                if len(optimized_locals) == len(formatted_output.local_questions):
                    formatted_output.global_question = optimized_global
                    formatted_output.local_questions = optimized_locals
                    logger.info("Template quality optimization successful")
                else:
                    logger.warning("Optimized local question count does not match, keeping the original version")
            else:
                logger.error(f"Template optimization failed: {response}")
                
        except Exception as e:
            logger.error(f"Error occurred during template optimization: {e}")
        
        return formatted_output
    
    def _extract_process_content(self, response: str) -> str:
        """
        Extract the content inside the <process> tag from the response
        
        Args:
            response: LLM response
            
        Returns:
            str: Extracted content, or the original response if not found
        """
        
        # Try to extract content from the <process></process> tag
        process_pattern = r'<process>\s*(.*?)\s*</process>'
        process_matches = re.findall(process_pattern, response, re.DOTALL)
        
        if process_matches:
            return process_matches[0].strip()
        
        # If no process tag is found, return the original response
        return response.strip()
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from the response
        
        Args:
            response: LLM response
            
        Returns:
            Dict[str, Any]: Parsed JSON data
        """
        
        # First try to extract JSON from the <process></process> tag
        process_pattern = r'<process>\s*({.*?})\s*</process>'
        process_matches = re.findall(process_pattern, response, re.DOTALL)
        
        if process_matches:
            try:
                return json.loads(process_matches[0])
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON extracted from process tag: {e}, content: {process_matches[0][:200]}...")
                try:
                    # Use json_repair to repair corrupted JSON
                    repaired_json = repair_json(process_matches[0])
                    logger.debug(f"JSON repair successful")
                    return json.loads(repaired_json)
                except Exception as e2:
                    logger.debug(f"JSON repair also failed: {e2}")
                    pass
        
        # Try direct parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.debug(f"Direct JSON parsing failed: {e}")
            try:
                # Use json_repair to repair corrupted JSON
                repaired_json = repair_json(response)
                logger.debug(f"JSON repair successful")
                return json.loads(repaired_json)
            except Exception as e2:
                logger.debug(f"JSON repair failed: {e2}")
                pass
        
        # Try to extract a JSON code block
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON extracted from code block: {e}")
                try:
                    # Use json_repair to repair corrupted JSON
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
                    # Use json_repair to repair corrupted JSON
                    repaired_json = repair_json(match)
                    return json.loads(repaired_json)
                except Exception:
                    continue
        
        # If all attempts fail, return an empty dictionary
        logger.warning(f"Unable to extract JSON from response: {response[:100]}...")
        return {}

    def _adversarial_audit(self, global_question: str, original_query: str) -> Dict[str, Any]:
        """
        Adversarially audit the global question
        
        Args:
            global_question: Global question
            original_query: Original query
            
        Returns:
            Dict[str, Any]: Audit result
        """
        logger.info("Starting adversarial audit")
        
        # Build the audit prompt
        audit_prompt = f"""
    Please audit the following global question:
    Original query: {original_query}
    Global question: {global_question}
    
    Please return the audit result in the required format.
    """
        
        messages = [
            {"role": "system", "content": ADVERSARIAL_AUDIT_PROMPT},
            {"role": "user", "content": audit_prompt}
        ]
        
        response, success = self.llm_manager.call_llm(messages)
        
        if success:
            try:
                # Extract the JSON result
                audit_data = self._extract_process_content(response)
                result_json = json.loads(audit_data)
                
                # Normalize the result format
                return {
                    "result": result_json.get("audit_result", "fail"),
                    "confidence": result_json.get("confidence", 0.0),
                    "issues_found": result_json.get("issues_found", []),
                }
                
            except Exception as e:
                logger.error(f"Failed to parse audit result: {e}")
                return {
                    "result": "fail",
                    "confidence": 0.0,
                    "issues_found": [f"Parsing failed: {str(e)}"],
                }
        else:
            logger.error(f"Adversarial audit failed: {response}")
            return {
                "result": "fail",
                "confidence": 0.0,
                "issues_found": ["LLM call failed"],
            }
