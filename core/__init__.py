"""Core modules for entity extraction and image synthesis."""

from .llm_client import LLMClient, llm_client
from .entity_extractor import (
    EntityExtractor,
    ImagePromptComposer, 
    generate_composite_image_spec,
    ENTITY_EXTRACTION_PROMPT,
    IMAGE_GENERATION_PROMPT
)
from .vlm_recognizer import VLMRecognizer, RecognitionGate, vlm_recognizer, recognition_gate
from .text_rewriter import TextRedactor, AdversarialRewriter, TextProcessor, text_processor
from .payload_assembler import PayloadAssembler, MultiModalJailbreaker, payload_assembler, multimodal_jailbreaker
from .entity_migration import (
    mask_entities_in_text,
    vlm_locate_elements_in_image,
    fill_masks_with_locations,
    polish_text_coherence,
    migrate_entities_to_image
)

__all__ = [
    "LLMClient",
    "llm_client",
    "EntityExtractor",
    "ImagePromptComposer",
    "generate_composite_image_spec", 
    "ENTITY_EXTRACTION_PROMPT",
    "IMAGE_GENERATION_PROMPT",
    "VLMRecognizer",
    "RecognitionGate",
    "vlm_recognizer",
    "recognition_gate",
    "TextRedactor",
    "AdversarialRewriter", 
    "TextProcessor",
    "text_processor",
    "PayloadAssembler",
    "MultiModalJailbreaker",
    "payload_assembler",
    "multimodal_jailbreaker",
    "mask_entities_in_text",
    "vlm_locate_elements_in_image",
    "fill_masks_with_locations",
    "polish_text_coherence",
    "migrate_entities_to_image"
]
