# -*- coding: utf-8 -*-
"""
Data models module - defines all data classes and structures

Contains:
- EntityRefinementResult: Entity refinement result
- RiddleResult: Knowledge-seeking question result
- QualityAssessment: Quality assessment
- FormattedOutput: Formatted output
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class EntityRefinementResult:
    """Entity refinement result data class"""
    original_entity: str
    refined_entity: str
    confidence: float
    refinement_type: str  # "specific", "euphemism", "abstraction"


@dataclass
class EntityDetectionResult:
    """Entity detection result data class"""
    detected_entities: List['EntityAnalysis']
    processing_complete: bool = True
    detection_method: str = "standard"
    confidence_score: float = 1.0


@dataclass
class RiddleResult:
    """Knowledge-seeking question result data class"""
    entity_name: str
    entity_type: str  # Stores the original Entity type: "target_object", "action", "concept"
    riddle_question: str
    riddle_answer: str
    quality_score: float
    is_safe: bool
    need_riddle: bool
    construction_reasoning: str
    quality_assessment: Optional['QualityAssessment'] = None


@dataclass
class QualityAssessment:
    """Quality assessment data class"""
    uniqueness_score: float
    difficulty_score: float
    safety_score: float
    fluency_score: float
    completeness_score: float
    overall_score: float
    feedback: str
    improvement_suggestions: Optional[List[str]] = None


@dataclass
class EntityQuestionMapping:
    """Entity-question mapping data class"""
    entity_name: str
    entity_type: str
    knowledge_seeking: str
    quality_score: float


@dataclass
class FormattedOutput:
    """Formatted output data class"""
    global_question: str
    local_questions: List[str]  # Keep for backward compatibility
    local_entities: List[str]   # Keep for backward compatibility
    entity_question_mappings: List[EntityQuestionMapping]  # Added: explicit Entity-question mappings
    formatted_output: str


@dataclass
class EntityAnalysis:
    """Entity analysis data class"""
    entity_name: str
    entity_type: str  # "target_object", "action", "concept"
    danger_level: str  # "safe", "mild", "dangerous"
    priority: int  # 1-5, 5 is the most important
    need_riddle: bool
    detailed_analysis: Optional[Dict] = None
    confidence: Optional[float] = None
    analysis_reasoning: Optional[str] = None
    safety_concerns: Optional[List[str]] = None



@dataclass
class QualityDimension:
    """Quality dimension data class"""
    achieved: bool
    score: float
    reason: str
    improvement_suggestions: List[str]


@dataclass
class QualityCheckResult:
    """Quality check result data class"""
    round: int
    quality_score: float
    can_exit: bool
    blocking_issues: List[str]
    quality_dimensions: Dict[str, QualityDimension]
    scoring_reasons: Dict[str, Dict[str, str]]
    improvement_suggestions: List[str]
    current_best_output: str


@dataclass
class RoundState:
    """Round state data class"""
    round: int
    agent_thinking: Dict
    tools_executed: List[str]
    round_results: Dict
    termination_reason: Optional[str] = None
    termination_check: Optional[Dict] = None
    force_terminated: bool = False
    exit_reason: Optional[str] = None
    final_quality_score: Optional[float] = None
    quality_check: Optional[QualityCheckResult] = None


@dataclass
class QuestionMemory:
    """Question memory data class"""
    original_question: str
    current_round: int
    original_entities: List[str]
    entity_analysis: Dict[str, EntityAnalysis]
    riddle_candidates: List[Dict]
    quality_scores: List[Dict]
    round_history: List[RoundState]
    global_template: Dict
    final_decision: Optional[Dict] = None
    quality_progression: List[QualityCheckResult] = None


@dataclass
class ModelConfig:
    """Model configuration data class"""
    name: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0