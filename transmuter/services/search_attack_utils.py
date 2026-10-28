# -*- coding: utf-8 -*-
"""
Utility functions module - contains helper functions and utility methods

contains:
- Text processing utilities
- JSON parsing utilities
- Quality assessment utilities
- Logging and debugging utilities
- Configuration management utilities
"""

import logging
import json
import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utility class"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing extra whitespace characters
        
        Args:
            text: Input text
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace characters
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters (keep Chinese, English, digits, and basic punctuation)
        text = re.sub(r'[^\u4e00-\u9fff\w\s.,!?;:()\[\]{}"\'-]', '', text)
        
        return text
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract text keywords
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords
            
        Returns:
            List[str]: List of keywords
        """
        if not text:
            return []
        
        # Simple keyword extraction (based on word frequency)
        words = re.findall(r'[\u4e00-\u9fff\w]+', text.lower())
        
        # Filter stop words
        stop_words = {'\u7684', '\u4e86', '\u5728', '\u662f', '\u6211', '\u6709', '\u548c', '\u5c31', '\u4e0d', '\u4eba', '\u90fd', '\u4e00', '\u4e00\u4e2a', '\u4e0a', '\u4e5f', '\u5f88', '\u5230', '\u8bf4', '\u8981', '\u53bb', '\u4f60', '\u4f1a', '\u7740', '\u6ca1\u6709', '\u770b', '\u597d', '\u81ea\u5df1', '\u8fd9'}
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:max_keywords]]
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts (simple Jaccard similarity)
        
        Args:
            text1: Text 1
            text2: Text 2
            
        Returns:
            float: Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # Extract word sets
        words1 = set(re.findall(r'[\u4e00-\u9fff\w]+', text1.lower()))
        words2 = set(re.findall(r'[\u4e00-\u9fff\w]+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """
        Truncate text to the specified length
        
        Args:
            text: Input text
            max_length: Maximum length
            suffix: Suffix
            
        Returns:
            str: Truncated text
        """
        if not text or len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix

class JSONProcessor:
    """JSON processing utility class"""
    
    @staticmethod
    def safe_json_loads(json_str: str, default: Any = None) -> Any:
        """
        Safe JSON parsing
        
        Args:
            json_str: JSON string
            default: Default value
            
        Returns:
            Any: Parsing result or the default value
        """
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"JSON parsing failed: {e}")
            return default
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extract a JSON object from text
        
        Args:
            text: Input text
            
        Returns:
            Optional[Dict[str, Any]]: Extracted JSON object
        """
        if not text:
            return None
        
        # Try direct parsing
        result = JSONProcessor.safe_json_loads(text)
        if result is not None:
            return result
        
        # Try to extract a JSON code block
        json_pattern = r'```(?:json)?\s*({.*?})\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            result = JSONProcessor.safe_json_loads(match)
            if result is not None:
                return result
        
        # Try to find a JSON object
        json_pattern = r'{[^{}]*(?:{[^{}]*}[^{}]*)*}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            result = JSONProcessor.safe_json_loads(match)
            if result is not None:
                return result
        
        return None
    
    @staticmethod
    def format_json(data: Any, indent: int = 2, ensure_ascii: bool = False) -> str:
        """
        Format JSON data
        
        Args:
            data: Data to format
            indent: Number of indentation spaces
            ensure_ascii: Whether to ensure ASCII encoding
            
        Returns:
            str: Formatted JSON string
        """
        try:
            return json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON formatting failed: {e}")
            return str(data)

class QualityMetrics:
    """Quality assessment utility class"""
    
    @staticmethod
    def calculate_weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
        """
        Calculate the weighted score
        
        Args:
            scores: Score dictionary
            weights: Weight dictionary
            
        Returns:
            float: Weighted score
        """
        if not scores or not weights:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for key, score in scores.items():
            if key in weights:
                weight = weights[key]
                total_weighted_score += score * weight
                total_weight += weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    @staticmethod
    def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Normalize the score to the specified range
        
        Args:
            score: Original score
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            float: Normalized score
        """
        return max(min_val, min(max_val, score))
    
    @staticmethod
    def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate the confidence interval
        
        Args:
            scores: List of scores
            confidence: Confidence level
            
        Returns:
            Tuple[float, float]: Confidence interval lower and upper bounds
        """
        if not scores:
            return 0.0, 0.0
        
        import statistics
        
        mean = statistics.mean(scores)
        if len(scores) == 1:
            return mean, mean
        
        stdev = statistics.stdev(scores)
        n = len(scores)
        
        # Simplified confidence interval calculation (assuming a normal distribution)
        margin = 1.96 * (stdev / (n ** 0.5))  # z-value for 95% confidence
        
        return mean - margin, mean + margin

class PerformanceMonitor:
    """Performance monitoring utility class"""
    
    def __init__(self):
        self.start_times = {}
        self.performance_data = []
    
    def start_timer(self, operation: str):
        """
        Start timing
        
        Args:
            operation: Operation name
        """
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """
        End timing and return the elapsed time
        
        Args:
            operation: Operation name
            
        Returns:
            float: Elapsed time (seconds)
        """
        if operation not in self.start_times:
            logger.warning(f"Operation {operation} has no start timer")
            return 0.0
        
        elapsed = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        # Record performance data
        self.performance_data.append({
            "operation": operation,
            "elapsed_time": elapsed,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Operation {operation} took: {elapsed:.3f} seconds")
        return elapsed
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get the performance summary
        
        Returns:
            Dict[str, Any]: Performance summary
        """
        if not self.performance_data:
            return {}
        
        operations = {}
        for record in self.performance_data:
            op = record["operation"]
            elapsed = record["elapsed_time"]
            
            if op not in operations:
                operations[op] = []
            operations[op].append(elapsed)
        
        summary = {}
        for op, times in operations.items():
            summary[op] = {
                "count": len(times),
                "total_time": sum(times),
                "average_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times)
            }
        
        return summary

class ConfigManager:
    """Configuration management utility class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config_data = {}
        
        if config_file and Path(config_file).exists():
            self.load_config()
    
    def load_config(self, config_file: Optional[str] = None) -> bool:
        """
        Load the configuration file
        
        Args:
            config_file: Configuration file path
            
        Returns:
            bool: Whether loading succeeded
        """
        file_path = config_file or self.config_file
        if not file_path:
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.config_data = json.load(f)
            logger.info(f"Configuration file loaded successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load configuration file: {e}")
            return False
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """
        Save the configuration file
        
        Args:
            config_file: Configuration file path
            
        Returns:
            bool: Whether saving succeeded
        """
        file_path = config_file or self.config_file
        if not file_path:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration file saved successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration file: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value
        
        Args:
            key: Configuration key (supports dot-separated nested keys)
            default: Default value
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value
        
        Args:
            key: Configuration key (supports dot-separated nested keys)
            value: Configuration value
        """
        keys = key.split('.')
        config = self.config_data
        
        # Create nested structure
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value

class HashUtils:
    """Hashing utility class"""
    
    @staticmethod
    def generate_hash(data: Union[str, bytes], algorithm: str = 'md5') -> str:
        """
        Generate a hash value for the data
        
        Args:
            data: Data to hash
            algorithm: Hashing algorithm
            
        Returns:
            str: Hash value
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(data)
        return hash_obj.hexdigest()
    
    @staticmethod
    def generate_content_id(content: str) -> str:
        """
        Generate a unique ID for the content
        
        Args:
            content: Content string
            
        Returns:
            str: Content ID
        """
        # Generate a unique ID from the content and timestamp
        timestamp = str(time.time())
        combined = f"{content}_{timestamp}"
        return HashUtils.generate_hash(combined, 'sha256')[:16]

class ValidationUtils:
    """Validation utility class"""
    
    @staticmethod
    def validate_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
        """
        Validate whether the score is within the valid range
        
        Args:
            score: Score
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            bool: Whether it is valid
        """
        return isinstance(score, (int, float)) and min_val <= score <= max_val
    
    @staticmethod
    def validate_text(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
        """
        Validate whether the text is valid
        
        Args:
            text: Text
            min_length: Minimum length
            max_length: Maximum length
            
        Returns:
            bool: Whether it is valid
        """
        return isinstance(text, str) and min_length <= len(text.strip()) <= max_length
    
    @staticmethod
    def validate_list(lst: List[Any], min_items: int = 0, max_items: int = 1000) -> bool:
        """
        Validate whether the list is valid
        
        Args:
            lst: List
            min_items: Minimum number of items
            max_items: Maximum number of items
            
        Returns:
            bool: Whether it is valid
        """
        return isinstance(lst, list) and min_items <= len(lst) <= max_items

# Global utility instances
text_processor = TextProcessor()
json_processor = JSONProcessor()
quality_metrics = QualityMetrics()
performance_monitor = PerformanceMonitor()
hash_utils = HashUtils()
validation_utils = ValidationUtils()

# Convenience functions
def clean_and_truncate(text: str, max_length: int = 100) -> str:
    """Clean and truncate text"""
    cleaned = text_processor.clean_text(text)
    return text_processor.truncate_text(cleaned, max_length)

def safe_extract_json(text: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """Safely extract JSON"""
    result = json_processor.extract_json_from_text(text)
    return result if result is not None else (default or {})

def calculate_quality_score(scores: Dict[str, float], weights: Dict[str, float] = None) -> float:
    """Calculate the quality score"""
    if weights is None:
        weights = {key: 1.0 for key in scores.keys()}
    return quality_metrics.calculate_weighted_score(scores, weights)

def generate_unique_id(content: str = "") -> str:
    """Generate a unique ID"""
    if content:
        return hash_utils.generate_content_id(content)
    else:
        return hash_utils.generate_hash(str(time.time()))

def log_performance(operation: str):
    """Performance monitoring decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.end_timer(operation)
        return wrapper
    return decorator

def format_output_to_tags(formatted_output) -> str:
    """
    Convert a FormattedOutput object into a string in <GLOBAL> and <LOCAL> tag format
    
    Args:
        formatted_output: FormattedOutput object or string
        
    Returns:
        str: Formatted tag string
    """
    # If it is already a string, return it directly
    if isinstance(formatted_output, str):
        if "<GLOBAL>" in formatted_output and "<LOCAL" in formatted_output:
            return formatted_output
    
    # If it is a FormattedOutput object, convert it
    if hasattr(formatted_output, 'global_question'):
        result_parts = []
        
        # Build the global question with local questions embedded
        global_question = formatted_output.global_question
        
        # Get the Entity question mappings
        entity_mappings = []
        if hasattr(formatted_output, 'entity_question_mappings') and formatted_output.entity_question_mappings:
            entity_mappings = formatted_output.entity_question_mappings
        elif hasattr(formatted_output, 'local_questions') and formatted_output.local_questions:
            # Create simple mappings for compatibility
            for question in formatted_output.local_questions:
                entity_mappings.append(type('obj', (object,), {'knowledge_seeking': question})())
        
        # Apply intelligent EntityType filtering, keeping only Entities of target_object type
        filtered_mappings = []
        for mapping in entity_mappings:
            # Check EntityType, keep only target objects (technical Entities), and filter out actions and abstract concepts
            if hasattr(mapping, 'entity_type') and mapping.entity_type == 'target_object':
                # Strictly filter by entity_type, without name fallback
                filtered_mappings.append(mapping)
            elif not hasattr(mapping, 'entity_type'):
                # Strict requirement: raise an error directly for mappings missing the entity_type field, with no fallback
                raise ValueError(f"Entity mapping is missing the entity_type field, unable to construct knowledge-solving question: {getattr(mapping, 'entity_name', 'unknown')}")
        
        # Build the global question with local questions and answers embedded as tags
        if filtered_mappings:
            # Build the LOCAL and ANSWER tag parts
            local_answer_tags = []
            for i, mapping in enumerate(filtered_mappings, 1):
                local_answer_tags.append(f"<LOCAL_{i}>{mapping.knowledge_seeking}</LOCAL_{i}><ANSWER_{i}>{mapping.entity_name}</ANSWER_{i}>")
            
            # Use a regular expression to match the end of the first question (there may be spaces or other characters after the question mark)
            question_pattern = r'^(.*?[\uff1f?])\s*(.*)$'
            match = re.match(question_pattern, global_question, re.DOTALL)
            
            if match:
                # Successfully matched the question pattern
                first_question = match.group(1).strip()
                remaining_content = match.group(2).strip()
                
                # Check whether the first question duplicates any LOCAL question
                first_question_clean = first_question.rstrip('\uff1f?').strip()
                is_duplicate = any(
                     first_question_clean in mapping.knowledge_seeking or
                     mapping.knowledge_seeking in first_question_clean 
                     for mapping in filtered_mappings
                 )
                
                if is_duplicate:
                    # If duplicated, use only the LOCAL tags and the remaining content
                    global_with_locals = f"<GLOBAL>{''.join(local_answer_tags)}{remaining_content}</GLOBAL>"
                else:
                    # If not duplicated, keep the first question
                    global_with_locals = f"<GLOBAL>{first_question}{''.join(local_answer_tags)}{remaining_content}</GLOBAL>"
                
                result_parts.append(global_with_locals)
            else:
                # Regex matching failed, regeneration is required
                logger.error(f"Regex matching failed, unable to parse the question format, regeneration required: {global_question}")
                raise ValueError(f"Question format does not meet requirements, unable to match a question format ending with a question mark, regeneration required: {global_question}")
        else:
            # If there are no local questions, add the global question directly
            result_parts.append(f"<GLOBAL>{global_question}</GLOBAL>")
        
        return "\n".join(result_parts)
    
    # If none of the above, return the string representation
    return str(formatted_output)