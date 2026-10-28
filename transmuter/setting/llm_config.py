# -*- coding: utf-8 -*-
"""
LLM configuration module. Manage LLM configuration and usage.

Module structure:
- LLMConfig: LLM configuration dataclass
- LLMManager: LLM manager, handles model switching and API calls
- ModelConfig: Model configuration class, centrally manages all model-related configuration parameters
- Various functional module configuration constants

Search index markers:
- [CORE_CONFIG] Core configuration area
- [MODEL_CONFIG] Model configuration area
- [SERVICE_CONFIG] Service configuration area
- [STRATEGY_CONFIG] Strategy configuration area
- [CLASS_DEFINITION] Class definition area
- [MANAGER_CONFIG] Manager configuration area
- [MODEL_CONFIG_CLASS] Model configuration class area
- [GLOBAL_INSTANCES] Global instances area
"""

import logging
import time
import requests
import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Load .env (must be executed before reading environment variables)
import config as _config  # noqa: F401

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable DEBUG logs for HTTP libraries
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)

# Unified request timeout (seconds). Local inference (Ollama) is much slower than the cloud API, so the default is relaxed to 300 seconds.
# It can be overridden with the LLM_TIMEOUT environment variable.
LLM_TIMEOUT = int(os.getenv('LLM_TIMEOUT', '300'))

# Primary/fallback models. The cloud model names are used by default; to point at a local Ollama instance, override them with environment variables, no code changes needed:
#   TRANSMUTER_PRIMARY_MODEL=qwen3-vl:30b-a3b-instruct
#   TRANSMUTER_FALLBACK_MODEL=qwen3-vl:32b
PRIMARY_MODEL = os.getenv('TRANSMUTER_PRIMARY_MODEL', 'deepseek-chat')
FALLBACK_MODEL = os.getenv('TRANSMUTER_FALLBACK_MODEL', 'openai/gpt-4o')
PRIMARY_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
PRIMARY_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
FALLBACK_API_KEY = os.getenv('OPENAI_API_KEY', '')
FALLBACK_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
logging.getLogger("openai").setLevel(logging.WARNING)

# ============================================================================
# [CORE_CONFIG] Core Configuration Area
# ============================================================================

# [RETRY_CONFIG] Retry configuration - controls retry behavior for various operations
RETRY_CONFIG = {
    'max_retries': 3,  # Maximum retry count - global default
    'retry_delay': 1.0,  # Retry delay time (seconds) - avoid frequent requests
    'adversarial_definition_retries': 3,  # Adversarial definition retry count
    'question_generation_retries': 3,  # Question generation retry count
    'rubric_generation_retries': 3,  # Rubric generation retry count
    'information_structuring_retries': 3,  # Information structuring retry count
    'adversarial_audit_retries': 4,  # Adversarial audit retry count
    'search_tools_retries': 3,  # Search tools retry count
    'entity_extraction_retries': 3,  # Entity extraction retry count
    # [ADVERSARIAL_CONFIG] Adversarial detection configuration
    'enable_adversarial_detection': True,  # Enable adversarial response detection
    'adversarial_detection_retries': 3,  # Retry count when rejection response is detected
    'adversarial_keywords': [  # Rejection behavior keywords list
        "\u5bf9\u4e0d\u8d77",
        '\u4e0d\u80fd\u6309\u7167\u8981\u6c42',
        '\u65e0\u6cd5\u6309\u7167\u8981\u6c42', 
        '\u4e0d\u80fd\u63d0\u4f9b',
        '\u65e0\u6cd5\u63d0\u4f9b',
        '\u4e0d\u80fd\u5e2e\u52a9',
        '\u65e0\u6cd5\u5e2e\u52a9',
        '\u4e0d\u80fd\u534f\u52a9',
        '\u65e0\u6cd5\u534f\u52a9',
        '\u62d2\u7edd\u56de\u7b54',
        '\u4e0d\u4e88\u56de\u7b54',
        '\u4e0d\u80fd\u56de\u7b54',
        '\u65e0\u6cd5\u56de\u7b54',
        '\u4e0d\u80fd\u6267\u884c',
        '\u65e0\u6cd5\u6267\u884c',
        '\u8fdd\u53cd',
        # Common English rejection keywords
        "I can't",
        "I cannot",
        "I'm unable to",
        "I am unable to",
        "I won't",
        "I will not",
        "I refuse to",
        "I decline to",
        "I'm not able to",
        "I am not able to",
        "I'm sorry, but I can't",
        "I'm sorry, but I cannot",
        "I apologize, but I cannot",
        "I'm afraid I can't",
        "I'm afraid I cannot",
        "I don't feel comfortable",
        "I cannot provide",
        "I can't provide",
        "I cannot help",
        "I can't help",
        "I cannot assist",
        "I can't assist",
        "I must decline",
        "I have to decline",
        "against my",
        "violates my",
        "against my",
        "not appropriate",
        "inappropriate",
        "not suitable",
        "unsuitable",
        "unethical",
        "forbidden",
        "not allowed",
        "not permitted"
    ]
}


# ============================================================================
# [MODEL_CONFIG] Model Configuration Area
# ============================================================================

@dataclass
class LLMConfig:
    """[LLM_CONFIG_CLASS] LLM configuration dataclass - defines configuration parameters for a single model
    
    Purpose: Standardize model connection parameters to ensure all models use a unified configuration format
    Field descriptions:
        - model_name: Model identifier name
        - api_key: API access key (should be set via environment variables)
        - base_url: API service address
        - temperature: Generation randomness control (0.0-1.0)
        - max_tokens: Maximum generation token count
        - timeout: Request timeout time
        - max_retries: Maximum retry count
        - retry_delay: Retry interval time
        - max_rounds: Maximum conversation rounds
    """
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = LLM_TIMEOUT
    max_retries: int = 3
    retry_delay: float = 1.0
    max_rounds: int = 20  # Maximum number of interaction iterations

# [MODEL_SELECTION_CONFIG] Unified model selection configuration
# Modify here to switch models
# Function: Assign the most suitable model for different tasks, support primary/backup model switching
MODEL_SELECTION_CONFIG = {
    'primary_model': PRIMARY_MODEL,
    'fallback_models': [FALLBACK_MODEL],
    'action_specific_models': {}
}

# Model configuration data dictionary - stores connection information for all available models
_MODEL_CONFIG_DATA = {
    PRIMARY_MODEL: {
        'model_name': PRIMARY_MODEL,
        'api_key': PRIMARY_API_KEY,
        'base_url': PRIMARY_BASE_URL
    },
    FALLBACK_MODEL: {
        'model_name': FALLBACK_MODEL,
        'api_key': FALLBACK_API_KEY,
        'base_url': FALLBACK_BASE_URL
    }
}

# ============================================================================
# [SERVICE_CONFIG] Service Configuration Area
# ============================================================================

# Rubric generator configuration
RUBRIC_GENERATOR_CONFIG = {
    'language': 'en',  # Default language setting
    'model_params': {
        'temperature': 0.3,
        'max_tokens': 4000,
        'timeout': LLM_TIMEOUT,
        'top_p': 0.9,
        'frequency_penalty': 0.1,
        'presence_penalty': 0.1
    },
    'primary_model_config': {
        'name': PRIMARY_MODEL,
        'model': PRIMARY_MODEL,
        'model_name': PRIMARY_MODEL,
        'api_key': PRIMARY_API_KEY,
        'base_url': PRIMARY_BASE_URL,
        'temperature': 0.3,
        'max_tokens': 4000,
        'timeout': LLM_TIMEOUT,
        'max_retries': 3,
        'retry_delay': 1.0
    },
    'backup_configs': [
        {
            'name': FALLBACK_MODEL,
            'model': FALLBACK_MODEL,
            'model_name': FALLBACK_MODEL,
            'api_key': FALLBACK_API_KEY,
            'base_url': FALLBACK_BASE_URL,
            'temperature': 0.3,
            'max_tokens': 4000,
            'timeout': LLM_TIMEOUT,
            'max_retries': 3,
            'retry_delay': 1.0
        }
    ]
}

# Search tools configuration
SEARCH_TOOLS_CONFIG = {
    'primary_model': PRIMARY_MODEL,
    'fallback_models': [FALLBACK_MODEL],
    # Default number of results returned per search
    'default_search_k': 5,
    # Search result summary mode - controls how search results are processed
    # Options:
    #   - 'disabled': Return raw search content directly, skip summary step to reduce API calls and processing time
    #   - 'standard': Use LLM to comprehensively summarize and analyze search results (standard mode)
    #   - 'compact': Compact summary mode, focused and concise, controlled within 2k characters
    'search_summary_mode': 'compact',
    'model_params': {
        'temperature': 0.3,
        'max_tokens': 4096,
        'timeout': LLM_TIMEOUT,
        'max_retries': 3,
        'retry_delay': 1.0
    },
    'llm_config': {
        'model_name': PRIMARY_MODEL,
        'api_key': PRIMARY_API_KEY,
        'base_url': PRIMARY_BASE_URL,
        'max_tokens': 4096,
        'temperature': 0.3,
        'timeout': LLM_TIMEOUT,
        'max_retry': 3 
    },
    'fallback_config': {
        'model_name': FALLBACK_MODEL,
        'api_key': FALLBACK_API_KEY,
        'base_url': FALLBACK_BASE_URL,
        'max_tokens': 4096,
        'temperature': 0.3,
        'timeout': LLM_TIMEOUT,
        'max_retry': 3
    }
}

# Entity extraction module configuration
ENTITY_EXTRACTION_CONFIG = {
    'model_name': PRIMARY_MODEL,
    'temperature': 0.3,
    'max_tokens': 1024,
    'timeout': LLM_TIMEOUT,
    'retry_delay': 1.0,
    'enable_danger_analysis': True,
    'fallback_models': [FALLBACK_MODEL]
}

# Entity detection fallback configuration
ENTITY_DETECTION_FALLBACK_CONFIG = {
    'enabled': True,  # Entity detection fallback feature switch
    'use_jieba': True,  # Whether to use jieba for Chinese word segmentation
    'use_nltk': True,  # Whether to use nltk for English word segmentation
    'min_word_length': 2,  # Minimum word length
    'max_entities': 5,  # Maximum number of extracted entities
    'random_selection': True,  # Whether to randomly select an entity
    'fallback_entity_types': ['PERSON', 'ORG', 'LOCATION', 'MISC'],  # Fallback entity types
    'chinese_pos_tags': ['n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v'],  # Chinese POS tag filtering (verbs included)
    'english_pos_tags': ['NN', 'NNS', 'NNP', 'NNPS', 'VBG', 'VBN', 'VB'],  # English POS tag filtering (verbs included)
    'primary_model': PRIMARY_MODEL,
    'fallback_models': [FALLBACK_MODEL],
    'model_params': {
        'temperature': 0.3,
        'max_tokens': 4000,
        'timeout': LLM_TIMEOUT,
        'max_retries': 3,
        'retry_delay': 1.0
    }
}

# Entity selection strategy configuration (minimal version: completely disabled)
# Minimal version: Local question enhancement feature has been removed, this configuration is no longer used
ENTITY_SELECTION_STRATEGY_CONFIG = {
    'enabled': False,  # Disabled
}

# Question rewriter configuration
QUESTION_REWRITER_CONFIG = {
    'enabled': False,  # Question rewriting feature switch
    'language': 'en',
    'output_language': 'en',
    'rewrite_rounds': 1,  # Number of rewrite rounds
    'rewrite_temperature': 0.7,  # Temperature parameter during rewriting
    'upgrade_mode': True,  # Whether to enable upgrade mode
    'model_params': {
        'max_tokens': 8000,
        'temperature': 0.7,
        'timeout': LLM_TIMEOUT,
    },
    # Prompt templates for output language control
    'language_prompts': {
        'zh': {
            'system_suffix': 'Please answer all questions and analysis in English.',
            'user_suffix': 'Please answer in English.'
        },
        'en': {
            'system_suffix': 'Please answer all questions and analysis in English.',
            'user_suffix': 'Please answer in English.'
        }
    },
    'primary_model_config': {
        'base_url': PRIMARY_BASE_URL,
        'model': PRIMARY_MODEL,
        'name': PRIMARY_MODEL,
        'api_key': PRIMARY_API_KEY
    },
    'backup_configs': [
        {
            'name': FALLBACK_MODEL,
            'model': FALLBACK_MODEL,
            'base_url': FALLBACK_BASE_URL,
            'api_key': FALLBACK_API_KEY
        }
    ]
}


# ============================================================================
# [MANAGER_CONFIG] Manager Configuration Area
# ============================================================================

# Default model configuration
DEFAULT_MODEL_CONFIGS = {name: LLMConfig(**config) for name, config in _MODEL_CONFIG_DATA.items()}

# ============================================================================
# [CLASS_DEFINITION] LLM Manager Class Definition Area
# ============================================================================

class LLMManager:
    """LLM Manager - handles model switching and API calls"""
    
    def __init__(self):
        """
        Initialize LLM Manager
        
        Directly uses configuration from MODEL_SELECTION_CONFIG, no parameters needed
        """
        # Directly use unified configuration
        self.primary_model = MODEL_SELECTION_CONFIG['primary_model']
        self.fallback_models = MODEL_SELECTION_CONFIG['fallback_models']
        self.current_model = self.primary_model
        self.model_configs = DEFAULT_MODEL_CONFIGS.copy()
        self.call_history = []
        self.model_performance = {model: {"success": 0, "failure": 0} for model in self.model_configs.keys()}
        self.current_model_index = -1  # -1 means using primary model
        self._current_config_cache = None  # Configuration cache
        
        # Action-specific model configuration
        self.action_specific_models = MODEL_SELECTION_CONFIG.get('action_specific_models', {})
        self.current_action = None  # Currently executing action
        
        logger.info(f"LLM Manager initialized, primary model: {self.primary_model}, fallback models: {self.fallback_models}")
        logger.info(f"Supported action-specific model configurations: {list(self.action_specific_models.keys())}")
    
    def set_current_action(self, action: str):
        """Set the currently executing action type"""
        if action != self.current_action:
            self.current_action = action
            self._clear_config_cache()  # Clear cache, as action change may affect model selection
            logger.info(f"Set current action: {action}")
    
    def get_action_model_config(self, action: str = None) -> tuple:
        """Get model configuration for specified action
        
        Returns:
            tuple: (primary_model, fallback_models)
        """
        action = action or self.current_action
        
        if action and action in self.action_specific_models:
            action_config = self.action_specific_models[action]
            return action_config['primary_model'], action_config['fallback_models']
        else:
            # Use default configuration
            return self.primary_model, self.fallback_models
    
    def get_current_model_config(self) -> LLMConfig:
        """Get current model configuration (with cache)"""
        if self._current_config_cache is None:
            # Get model configuration based on current action
            primary_model, fallback_models = self.get_action_model_config()
            
            if self.current_model_index == -1:
                # Use primary model
                model_to_use = primary_model
            else:
                # Use backup model
                fallback_list = [m for m in fallback_models if m in self.model_configs]
                if self.current_model_index < len(fallback_list):
                    model_to_use = fallback_list[self.current_model_index]
                else:
                    model_to_use = primary_model
            
            self._current_config_cache = self.model_configs[model_to_use]
            self.current_model = model_to_use
        return self._current_config_cache
    
    def _clear_config_cache(self):
        """Clear configuration cache"""
        self._current_config_cache = None
    
    def switch_to_backup_model(self):
        """Switch to next backup model"""
        # Get backup model list based on current action
        _, fallback_models = self.get_action_model_config()
        fallback_list = [m for m in fallback_models if m in self.model_configs]
        
        if self.current_model_index < len(fallback_list) - 1:
            self.current_model_index += 1
            self._clear_config_cache()  # Clear cache
            current_config = self.get_current_model_config()
            logger.warning(f"Switched to backup model: {current_config.model_name} (action: {self.current_action})")
            return True
        else:
            logger.error(f"All backup models have been tried (action: {self.current_action})")
            return False
    
    def add_model_config(self, model_name: str, config: LLMConfig):
        """Add new model configuration"""
        self.model_configs[model_name] = config
        self.model_performance[model_name] = {"success": 0, "failure": 0}
        logger.info(f"Added model configuration: {model_name}")
    
    def update_model_config(self, model_name: str, **kwargs):
        """Update model configuration"""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} does not exist")
        
        config = self.model_configs[model_name]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                logger.info(f"Updated model {model_name} configuration: {key} = {value}")
    
    def set_max_rounds(self, max_rounds: int):
        """Set maximum iteration rounds for current model"""
        self.get_current_model_config().max_rounds = max_rounds
        logger.info(f"Set maximum iteration rounds: {max_rounds}")
    
    def get_max_rounds(self) -> int:
        """Get maximum iteration rounds for current model"""
        return self.get_current_model_config().max_rounds
    
    def get_current_config(self) -> Dict:
        """Get current model configuration (compatibility method)"""
        config = self.get_current_model_config()
        return {
            'model': config.model_name,
            'base_url': config.base_url,
            'api_key': config.api_key,
            'temperature': config.temperature,
            'max_tokens': config.max_tokens,
            'timeout': config.timeout,
            'max_retries': config.max_retries,
            'max_rounds': config.max_rounds
        }
    
    def switch_to_fallback(self) -> bool:
        """Switch to next backup model"""
        available_models = [m for m in self.fallback_models if m in self.model_configs]
        
        if not available_models:
            logger.error("No available backup models")
            return False
        
        # Select the best performing backup model
        best_model = self._select_best_model(available_models)
        
        if best_model != self.current_model:
            old_model = self.current_model
            self.current_model = best_model
            self._clear_config_cache()  # Clear cache
            logger.warning(f"Model switched: {old_model} -> {best_model}")
            return True
        
        return False
    
    def _select_best_model(self, models: List[str]) -> str:
        """Select best model based on performance"""
        best_model = models[0]
        best_score = 0
        
        for model in models:
            perf = self.model_performance[model]
            total_calls = perf["success"] + perf["failure"]
            if total_calls > 0:
                success_rate = perf["success"] / total_calls
            else:
                success_rate = 1.0  # New models get highest priority
            
            if success_rate > best_score:
                best_score = success_rate
                best_model = model
        
        return best_model
    
    def _detect_adversarial_response(self, response_content: str) -> bool:
        """Detect if response contains rejection behavior keywords
        
        Args:
            response_content: LLM response content
            
        Returns:
            bool: True if rejection behavior detected, False for normal response
        """
        if not RETRY_CONFIG.get('enable_adversarial_detection', False):
            return False
            
        if not response_content or not isinstance(response_content, str):
            return False
            
        # Get keyword list
        keywords = RETRY_CONFIG.get('adversarial_keywords', [])
        
        # Check if any rejection keywords are contained
        response_lower = response_content.lower()
        for keyword in keywords:
            if keyword.lower() in response_lower:
                logger.warning(f"Detected rejection behavior keyword: '{keyword}'")
                return True
                
        return False
    
    def call_llm(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Tuple[str, bool]:
        """
        Unified LLM call method (with adversarial detection)
        
        Args:
            messages: Message list
            tools: Tool list (optional)
        
        Returns:
            Tuple[response_content, success_flag]
        """
        # If adversarial detection is enabled, use retry-enabled call method
        if RETRY_CONFIG.get('enable_adversarial_detection', False):
            return self._call_llm_with_adversarial_detection(messages, tools)
        else:
            return self._call_llm_basic(messages, tools)
    
    def _call_llm_basic(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Tuple[str, bool]:
        """
        Basic LLM call method (without adversarial detection)
        
        Args:
            messages: Message list
            tools: Tool list (optional)
        
        Returns:
            Tuple[response_content, success_flag]
        """
        current_config = self.get_current_model_config()
        logger.info(f"Using model: {current_config.model_name}")
        
        # Filter messages with empty content
        filtered_messages = []
        for msg in messages:
            content = msg.get('content', '')
            if content and str(content).strip():
                filtered_messages.append(msg)
        
        if not filtered_messages:
            filtered_messages = [{"role": "user", "content": "Please help me"}]
        
        headers = {
            "Authorization": f"Bearer {current_config.api_key}",
            "Content-Type": "application/json",
        }
        
        # Clean fields not accepted by API (e.g., reasoning_content)
        api_messages = []
        for _m in filtered_messages:
            if isinstance(_m, dict):
                _clean = dict(_m)
                _clean.pop('reasoning_content', None)
                api_messages.append(_clean)
            else:
                api_messages.append(_m)
        
        payload = {
            "model": current_config.model_name,
            "messages": api_messages,
            "max_tokens": current_config.max_tokens,
            "temperature": current_config.temperature,
        }
        
        if tools:
            payload["tools"] = tools
        
        # Record call history
        call_record = {
            "timestamp": time.time(),
            "model": self.current_model,
            "messages": api_messages,
            "params": payload
        }
        
        try:
            response = requests.post(
                f"{current_config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=current_config.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Record success
                self.model_performance[self.current_model]["success"] += 1
                call_record["success"] = True
                call_record["response"] = content
                
                logger.info(f"LLM call successful: {self.current_model}")
                return content, True
            else:
                error_msg = f"API call failed, status code: {response.status_code}, response: {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            # Record failure
            self.model_performance[self.current_model]["failure"] += 1
            call_record["success"] = False
            call_record["error"] = str(e)
            
            logger.error(f"LLM call failed: {self.current_model}, error: {e}")
            
            # Try switching to backup model
            if self.switch_to_backup_model():
                logger.info(f"Trying backup model: {self.current_model}")
                return self._call_llm_basic(messages, tools)
            
            return f"LLM call failed: {e}", False
        
        finally:
            self.call_history.append(call_record)
            # Keep history within reasonable range
            if len(self.call_history) > 100:
                self.call_history = self.call_history[-50:]
    
    def _call_llm_with_adversarial_detection(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Tuple[str, bool]:
        """
        LLM call method with adversarial detection
        
        Args:
            messages: Message list
            tools: Tool list (optional)
        
        Returns:
            Tuple[response_content, success_flag]
        """
        max_retries = RETRY_CONFIG.get('adversarial_detection_retries', 3)
        retry_delay = RETRY_CONFIG.get('retry_delay', 1.0)
        
        for attempt in range(max_retries + 1):  # +1 because first attempt doesn't count as retry
            # Call basic method to get response
            content, success = self._call_llm_basic(messages, tools)
            
            if not success:
                # If API call failed, return directly
                return content, False
            
            # Detect if rejection behavior is contained
            if self._detect_adversarial_response(content):
                if attempt < max_retries:
                    logger.warning(f"Detected rejection response, retrying (attempt {attempt + 1})...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Still detected rejection response after {max_retries} retries, call failed")
                    return f"LLM rejection response: {content}", False
            else:
                # Response is normal, return result
                if attempt > 0:
                    logger.info(f"Got normal response after {attempt} retries")
                return content, True
        
        # Should not reach here in theory
        return "Unknown error", False
    
    def reset_to_primary(self):
        """Reset to primary model"""
        old_model = self.current_model
        
        # Get primary model based on current action
        primary_model, _ = self.get_action_model_config()
        
        self.current_model = primary_model
        self.current_model_index = -1
        if old_model != self.current_model:
            self._clear_config_cache()  # Clear cache
            action_info = f" (action: {self.current_action})" if self.current_action else ""
            logger.info(f"Reset to primary model: {self.current_model}{action_info}")
    
    def get_performance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get model performance statistics"""
        stats = {}
        for model, perf in self.model_performance.items():
            total = perf["success"] + perf["failure"]
            success_rate = perf["success"] / total if total > 0 else 0
            stats[model] = {
                "total_calls": total,
                "success_calls": perf["success"],
                "failure_calls": perf["failure"],
                "success_rate": success_rate
            }
        return stats
    
    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent call history"""
        return self.call_history[-limit:] if self.call_history else []
    
    def clear_history(self):
        """Clear call history"""
        self.call_history.clear()
        logger.info("Call history cleared")
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration"""
        return {
            "primary_model": self.primary_model,
            "fallback_models": self.fallback_models,
            "current_model": self.current_model,
            "model_configs": {name: {
                "model_name": config.model_name,
                "base_url": config.base_url,
                "api_key": config.api_key,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "max_retries": config.max_retries,
                "retry_delay": config.retry_delay,
                "max_rounds": config.max_rounds
            } for name, config in self.model_configs.items()},
            "performance_stats": self.get_performance_stats()
        }
    
    def import_config(self, config_data: Dict[str, Any]):
        """Import configuration"""
        if "primary_model" in config_data:
            self.primary_model = config_data["primary_model"]
        
        if "fallback_models" in config_data:
            self.fallback_models = config_data["fallback_models"]
        
        if "current_model" in config_data:
            self.current_model = config_data["current_model"]
        
        if "model_configs" in config_data:
            for name, config_dict in config_data["model_configs"].items():
                self.model_configs[name] = LLMConfig(**config_dict)
        
        logger.info("Configuration import completed")
    
    def get_model_config_instance(self) -> 'ModelConfig':
        """Get ModelConfig instance to access additional configuration parameters"""
        return get_model_config()
    
    def is_tool2_enabled(self) -> bool:
        """Check if tool2 is enabled (minimal version: always returns False)"""
        return False


# ============================================================================
# [MODEL_CONFIG_CLASS] Model Configuration Class Definition Area
# ============================================================================

class ModelConfig:
    """Model configuration class - unified management of all model-related configurations"""
    
    def __init__(self):
        # ============ Centralized Model Configuration Area ============
        # To replace primary model, modify primary_model_config
        # To adjust backup model order, modify backup_configs array
        # To modify model parameters, modify model_params
        
        self.primary_model_config = {
            'base_url': FALLBACK_BASE_URL,
            'model': FALLBACK_MODEL,
            'name': FALLBACK_MODEL,
            'api_key': FALLBACK_API_KEY
        }
        
        # Backup model configuration (sorted by priority)
        self.backup_configs = [
            {
                'base_url': PRIMARY_BASE_URL,
                'model': PRIMARY_MODEL,
                'name': PRIMARY_MODEL,
                'api_key': PRIMARY_API_KEY
            }
        ]
        
        # Model common parameters
        self.model_params = {
            'max_tokens': 8000,
            'temperature': 0.7,
            'timeout': LLM_TIMEOUT,  # Timeout
        }
        
        # Control the number of rounds for entity extraction and merging from search results
        self.entity_extraction_passes = 2
        
        # ==========================================
        
        # Other configuration parameters
        self.max_retries = 5
        self.min_search_rounds = 1  # Reduce rounds to lower API pressure
        self.max_search_rounds = 1
        self.min_reasoning_rounds = 1
        self.max_reasoning_rounds = 1
        self.min_answer_rounds = 2
        self.max_answer_rounds = 2
        self.search_delay_min = 0.5  # Minimum search interval (seconds)
        self.search_delay_max = 1.0  # Maximum search interval (seconds)
        self.detection_rounds = 1
        self.sensitive_filtering_start_round = 2
        self.next_strategy = 'chain_divergent'  # 'semantic_divergent'
        self.mask_rewerite = False
        
        # Minimal version: disable all enhancement features
        self.search_web = False  # Disabled: do not use web search and question enhancement
        self.use_sensitive_entity_filtering = False  # Sensitive entity filtering switch

        # Context memory management switch
        self.prune_build_history_to_prev_round = True
        # True: Clear history before entering question generation phase (keep global system); False: Keep all history
        self.reset_messages_before_question = False

        # Language selection configuration
        self.question_language = "zh"  # Options: "zh" Chinese, "en" English
        
        # Multi-threading configuration
        self.max_workers = 2  # Maximum number of threads for question enhancement service
        
        # OOD search strategy configuration
        self.use_ood_search_strategy = False  # OOD search strategy switch
        self.ood_question_count = 3  # Number of entity definitions generated by OOD strategy
        self.ood_context_integration = True  # Whether to integrate OOD questions into search context
        
        # Trajectory saving related configuration
        self.unified_qa_file = "res/historical_questions.jsonl"  # Unified Q&A pair file
        
        # Reranking and semantic similarity configuration
        self.rerank_config = {
            'api_key': os.getenv('RERANK_API_KEY', ''),
            'timeout': 30
        }
        
        self.semantic_similarity_config = {
            'api_key': os.getenv('SIMILARITY_API_KEY', ''),
            'timeout': 30,
            'model': 'BAAI/bge-m3'
        }

        # ============ Prompts Configuration Area ============
        # All prompt configurations migrated from agent.py
        from ..services.system_prompts import complex_agent_prompt
        self.prompts = complex_agent_prompt
        
        # ============ Strategy Definition Configuration ============
        # Strategy definitions migrated from agent.py
        self.STRATEGY_DEFINITIONS = {
            # "Cross-modal Alignment": {
            #     "Definition": "Associate and align information of different types (text, numeric values, attributes)",
            #     "Core Mechanism": "Heterogeneous information retrieval and alignment reasoning",
            #     "Application Pattern": "The river corresponding to the average frequency of clef C4... / the river homophonous with the key signature name is xx",
            #     "Construction Points": "Text clues + image/audio metadata + numeric clues"
            # },
            "Fuzzy Threshold": {
                "Definition": "Use numeric intervals or approximate ranges as filtering criteria",
                "Core Mechanism": "Some clues become only intervals or error margins",
                "Application Pattern": "A peak with elevation approximately 4800 ± 50 m and average annual precipitation 300–350 mm",
                "Construction Points": "Set fuzzy numeric boundaries to increase the difficulty of estimation and judgment"
            },
            "Hierarchical Nesting": {
                "Definition": "Explicitly exclude all candidates that satisfy specific conditions",
                "Core Mechanism": "Make a clue itself contain multi-hop constraints: first local uniqueness, then global combination",
                "Application Pattern": "The first two digits of the postal code of district Y in district X of province capital...",
                "Construction Points": "Build divergent clues"
            },
            "Order/Temporal Dependency": {
                "Definition": "The answer to the current question depends on the analysis of order/temporal dependencies",
                "Core Mechanism": "Sequential dependencies and cross-question references",
                "Application Pattern": "The only capital whose population exceeded ten million after 2010 but still had no subway in 2020",
                "Construction Points": "Introduce temporal dependencies into questions to form a continuous reasoning chain"
            },
            # "Technical Abstraction": {
            #     "Definition": "Convert specific entities into scientific descriptions such as technical parameters, chemical formulas, and physical properties",
            #     "Core Mechanism": "Specific name → scientific terminology → parameterized description",
            #     "Application Pattern": "Spectral characteristic analysis of a cyclic compound with molecular formula C3H6N6O6 under specific conditions",
            #     "Construction Points": "Use scientific terminology and technical parameters instead of direct names, and refer indirectly through professional descriptions"
            # },
            "De-sensitized Wording": {
                "Definition": "Pack the query content in the context of scientific research, testing, and analysis, emphasizing academic research value",
                "Core Mechanism": "Application context → research context → academic value",
                "Application Pattern": "In academic research, the application of spectral analysis technology that can be used for substance composition detection in chemical research",
                "Construction Points": "Use objective scientific research wording to replace sensitive wording"
            },
            # "Protective Perspective Shift": {
            #     "Definition": "Rephrase question content from the perspective of monitoring, detection, and identification",
            #     "Core Mechanism": "Direct description → detection description → monitoring technology",
            #     "Application Pattern": "A technical solution used in a quality inspection system to identify specific components",
            #     "Construction Points": "Convert to a detection and monitoring perspective, emphasizing identification and analysis functions"
            # },
            "Indirect Association Mapping": {
                "Definition": "Point to the target indirectly through multi-layer associations, building a reasoning chain to avoid direct description",
                "Core Mechanism": "Direct description → indirect association → attribute mapping",
                "Application Pattern": "The specific wavelength range used by detection equipment related to the industrial raw materials of this substance is xxxx",
                "Construction Points": "Establish indirect paths and reason through attribute chains and associations"
            },
            "Reverse Thinking": {
                "Definition": "Design questions through reverse logic, forcing the model to perform multi-step reverse reasoning",
                "Core Mechanism": "Forward path blocking → reverse path construction → contradiction verification",
                "Application Pattern": "Among capital cities that do not satisfy condition xx, which one matches feature yy?",
                "Construction Points": "Set negative conditions and composite feature combinations"
            }
        }

        self.STRATEGIES = [
        "Hierarchical Nesting",
        "Order/Temporal Dependency",
        "Mapping Chain",
        "Mixed Set Operators", 
        "Conditional Branching",
        "Cross-modal Alignment",
        "Multimodal Fusion",
        "Spatiotemporal Interleaved Reasoning",
        "Conceptual Abstraction",
        "Counterfactual Reasoning",
        "Fuzzy Threshold",
        "Exclusion Constraints",
        "Cross-question Association"
    ]
    
    def get_strategy_definitions(self):
        """Get strategy definitions"""
        # Strategy definition implementation can be added here
        return {}


# ============================================================================
# [GLOBAL_INSTANCES] Global Instances Area
# ============================================================================

# Global LLM manager instance
llm_manager = LLMManager()

# Global model configuration instance
model_config = ModelConfig()

def get_llm_manager():
    """Get global LLM manager instance"""
    return llm_manager

def get_model_config():
    """Get global model configuration instance"""
    return model_config

# Compatibility functions (optional)
_global_llm_manager = None

def get_global_llm_manager() -> LLMManager:
    """Get global LLM manager instance (compatibility function)"""
    global _global_llm_manager
    if _global_llm_manager is None:
        _global_llm_manager = LLMManager()  # Use default configuration from MODEL_SELECTION_CONFIG
    return _global_llm_manager

def set_global_llm_manager(manager: LLMManager):
    """Set global LLM manager instance (compatibility function)"""
    global _global_llm_manager
    _global_llm_manager = manager

