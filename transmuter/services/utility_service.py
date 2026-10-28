# -*- coding: utf-8 -*-
"""
Utility service module - provides common utility methods

contains:
- UtilityService: Utility service class
- JSON parsing methods
- Text processing methods
- Regular expression utilities
"""

import logging
import json
import re
from typing import Dict, Any, Optional, List
from json_repair import repair_json  # Import the json_repair library to repair corrupted JSON

# Configure logging
logger = logging.getLogger(__name__)

class UtilityService:
    """Utility service - provides common utility methods"""
    
    def __init__(self):
        """
        Initialize the utility service
        """
        logger.info("Tool service initialized")
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract JSON from the response
        
        Args:
            response: LLM response
            
        Returns:
            Dict[str, Any]: Parsed JSON data
        """
        logger.debug(f"Trying to extract JSON from response: {response[:100]}...")
        
        # First try to extract JSON from the <process></process> tag
        process_pattern = r'<process>\s*({.*?})\s*</process>'
        process_matches = re.findall(process_pattern, response, re.DOTALL)
        
        if process_matches:
            try:
                result = json.loads(process_matches[0])
                logger.debug("Successfully extracted JSON from <process> tag")
                return result
            except json.JSONDecodeError as e:
                try:
                    # Use json_repair to repair corrupted JSON
                    repaired_json = repair_json(process_matches[0])
                    result = json.loads(repaired_json)
                    logger.debug("Successfully extracted JSON from <process> tag (after repair)")
                    return result
                except Exception as repair_e:
                    logger.debug(f"JSON parsing in <process> tag failed, and repair also failed: {repair_e}")
        
        # Try to parse the entire response directly
        try:
            result = json.loads(response)
            logger.debug("Successfully parsed response as JSON directly")
            return result
        except json.JSONDecodeError as e:
            try:
                # Use json_repair to repair corrupted JSON
                repaired_json = repair_json(response)
                result = json.loads(repaired_json)
                logger.debug("Successfully parsed response as JSON directly (after repair)")
                return result
            except Exception as repair_e:
                logger.debug(f"Direct JSON parsing failed, and repair also failed: {repair_e}")
        
        # Try to extract a JSON code block
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            try:
                result = json.loads(matches[0])
                logger.debug("Successfully extracted JSON from code block")
                return result
            except json.JSONDecodeError as e:
                try:
                    # Use json_repair to repair corrupted JSON
                    repaired_json = repair_json(matches[0])
                    result = json.loads(repaired_json)
                    logger.debug("Successfully extracted JSON from code block (after repair)")
                    return result
                except Exception as repair_e:
                    logger.debug(f"JSON parsing in code block failed, and repair also failed: {repair_e}")
        
        # Try to find a JSON object (looser matching)
        json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
        matches = re.findall(json_pattern, response)
        
        for match in matches:
            try:
                result = json.loads(match)
                logger.debug("Successfully extracted JSON from the JSON object pattern")
                return result
            except json.JSONDecodeError:
                try:
                    # Use json_repair to repair corrupted JSON
                    repaired_json = repair_json(match)
                    result = json.loads(repaired_json)
                    logger.debug("Successfully extracted JSON from the JSON object pattern (after repair)")
                    return result
                except Exception:
                    continue
        
        # If all attempts fail, return an empty dictionary
        logger.warning(f"Unable to extract JSON from response: {response[:200]}...")
        return {}
    
    def extract_action_with_regex(self, response: str) -> Optional[str]:
        """
        Extract an action from the response using regular expressions
        
        Args:
            response: LLM response
            
        Returns:
            Optional[str]: Extracted action, or None if not found
        """
        logger.debug(f"Trying to extract action from response: {response[:100]}...")
        
        # Define action patterns
        action_patterns = [
            r'\*\*\u9009\u62e9\u52a8\u4f5c\*\*:\s*([^\n]+)',  # Match the "**Select action**: action_name" format
            r'\u9009\u62e9\u52a8\u4f5c:\s*([^\n]+)',  # Match the "Select action: action_name" format
            r'\*\*Action\*\*:\s*([^\n]+)',  # Match the "**Action**: action_name" format
            r'Action:\s*([^\n]+)',
            r'\*\*\u52a8\u4f5c\*\*:\s*([^\n]+)',  # Match the "**Action**: action_name" format
            r'\u52a8\u4f5c:\s*([^\n]+)',
            r'action["\']?\s*:\s*["\']([^"\'\']+)["\']?',
            r'"action"\s*:\s*"([^"]+)"',
            r'<action>([^<]+)</action>',
            r'\u6267\u884c\u52a8\u4f5c:\s*([^\n]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                raw_action = matches[0].strip()
                # Clean and normalize the action
                cleaned_action = self._clean_and_validate_action(raw_action)
                if cleaned_action:
                    logger.debug(f"Successfully extracted and cleaned action: {cleaned_action}")
                    return cleaned_action
        
        logger.debug("Unable to extract action from response")
        return None
    
    def _clean_and_validate_action(self, raw_action: str) -> Optional[str]:
        """
        Clean and validate the action
        
        Args:
            raw_action: Original action string
            
        Returns:
            Optional[str]: Cleaned valid action, or None if invalid
        """
        # Define the list of valid actions
        valid_actions = [
            "analyze_and_detect",
            "construct_and_assess", 
            "build_template",
            "terminate"
        ]
        
        # Clean the action string: remove special characters, extra spaces, etc.
        cleaned = re.sub(r'[^a-zA-Z_]', '', raw_action.lower().strip())
        
        # Check whether it matches a valid action
        for valid_action in valid_actions:
            if valid_action in cleaned or cleaned in valid_action:
                return valid_action
        
        # If there is no exact match, try fuzzy matching
        if 'analyze' in cleaned or 'detect' in cleaned:
            return "analyze_and_detect"
        elif 'construct' in cleaned or 'assess' in cleaned:
            return "construct_and_assess"
        elif 'build' in cleaned or 'template' in cleaned:
            return "build_template"
        elif 'terminate' in cleaned or 'end' in cleaned or 'finish' in cleaned:
            return "terminate"
        
        logger.warning(f"Unrecognized action: {raw_action}")
        return None
    
    def normalize_entity_type(self, entity_type: str) -> str:
        """
        Normalize EntityType
        
        Args:
            entity_type: Original EntityType
            
        Returns:
            str: Normalized EntityType
        """
        if not entity_type:
            return "concept"
        
        entity_type_lower = entity_type.lower().strip()
        
        # Action type keywords
        action_keywords = [
            'action', 'method', 'process', 'procedure', 'technique', 'approach',
            'strategy', 'way', 'manner', 'means', 'operation', 'function',
            '\u65b9\u6cd5', '\u6280\u672f', '\u8fc7\u7a0b', '\u7a0b\u5e8f', '\u7b56\u7565', '\u65b9\u5f0f', '\u624b\u6bb5', '\u64cd\u4f5c', '\u529f\u80fd',
            '\u52a8\u4f5c', '\u884c\u4e3a', '\u6d3b\u52a8', '\u5b9e\u8df5', '\u505a\u6cd5'
        ]
        
        # Check whether it contains action keywords
        for keyword in action_keywords:
            if keyword in entity_type_lower:
                logger.debug(f"EntityType '{entity_type}' normalized to 'action'")
                return "action"
        
        # Default to the concept type
        logger.debug(f"EntityType '{entity_type}' normalized to 'concept'")
        return "concept"
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and special characters
        
        Args:
            text: Original text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace characters
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special control characters
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
        
        return cleaned
    
    def extract_entities_from_text(self, text: str) -> List[str]:
        """
        Extract possible Entities from text (simple rule-based method)
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of extracted Entities
        """
        entities = []
        
        # Extract content inside quotes
        quoted_pattern = r'["\']([^"\']{2,})["\']'
        quoted_matches = re.findall(quoted_pattern, text)
        entities.extend(quoted_matches)
        
        # Extract proper nouns (phrases starting with a capital letter)
        proper_noun_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        proper_nouns = re.findall(proper_noun_pattern, text)
        entities.extend(proper_nouns)
        
        # Extract Chinese proper nouns (simple pattern)
        chinese_pattern = r'[\u4e00-\u9fff]{2,}'
        chinese_matches = re.findall(chinese_pattern, text)
        entities.extend(chinese_matches)
        
        # Deduplicate and filter
        unique_entities = list(set(entities))
        filtered_entities = [e for e in unique_entities if len(e.strip()) > 1]
        
        logger.debug(f"Extracted {len(filtered_entities)} potential Entities from text")
        return filtered_entities
    
    def is_valid_json(self, text: str) -> bool:
        """
        Check whether the text is valid JSON
        
        Args:
            text: Text to check
            
        Returns:
            bool: Whether it is valid JSON
        """
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            try:
                # Use json_repair to repair corrupted JSON
                repaired_json = repair_json(text)
                json.loads(repaired_json)
                return True
            except Exception:
                return False
    
    def extract_code_blocks(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Extract code blocks from text
        
        Args:
            text: Input text
            language: Specified language type (optional)
            
        Returns:
            List[str]: List of extracted code blocks
        """
        if language:
            pattern = f'```{language}\s*\n(.*?)\n```'
        else:
            pattern = r'```(?:\w+)?\s*\n(.*?)\n```'
        
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def format_entity_list(self, entities: List[str], max_length: int = 100) -> str:
        """
        Format the Entity list as a string
        
        Args:
            entities: Entity list
            max_length: Maximum length limit
            
        Returns:
            str: Formatted string
        """
        if not entities:
            return "None"
        
        formatted = ", ".join(entities)
        
        if len(formatted) > max_length:
            # Truncate and add an ellipsis
            truncated = formatted[:max_length-3] + "..."
            return truncated
        
        return formatted
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts (simple character-based method)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score (0.0-1.0)
        """
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and remove whitespace
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        if t1 == t2:
            return 1.0
        
        # Calculate character-level similarity
        set1 = set(t1)
        set2 = set(t2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def validate_entity_name(self, entity_name: str) -> bool:
        """
        Validate whether the Entity name is valid
        
        Args:
            entity_name: Entity name
            
        Returns:
            bool: Whether it is valid
        """
        if not entity_name or not isinstance(entity_name, str):
            return False
        
        # Remove whitespace characters
        cleaned = entity_name.strip()
        
        # Check the length
        if len(cleaned) < 1 or len(cleaned) > 100:
            return False
        
        # Check whether it contains only whitespace or special characters
        if not re.search(r'[\w\u4e00-\u9fff]', cleaned):
            return False
        
        return True
    
    def extract_numbers_from_text(self, text: str) -> List[float]:
        """
        Extract numbers from text
        
        Args:
            text: Input text
            
        Returns:
            List[float]: List of extracted numbers
        """
        # Match integers and decimals
        number_pattern = r'-?\d+(?:\.\d+)?'
        matches = re.findall(number_pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue
        
        return numbers