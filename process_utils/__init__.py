"""Utilities for parsing <process> blocks and validating schemas.

Modules:
- process_parser: Regex-based extractor for <process> ... </process> blocks.
- schemas: Typed validators for CompositeImageSpec, VLMRecognitionSpec,
  RedactionMappingSpec, RewrittenTextSpec, and PayloadMessageSpec.
"""

from .process_parser import ProcessBlock, extract_process_blocks, get_blocks_by_name, load_block_json
from .schemas import (
    PROCESS_SCHEMAS,
    validate_schema,
    validate_composite_image_spec,
    validate_vlm_recognition_spec,
    validate_redaction_mapping_spec,
    validate_rewritten_text_spec,
    validate_payload_message_spec,
)

__all__ = [
    "ProcessBlock",
    "extract_process_blocks",
    "get_blocks_by_name",
    "load_block_json",
    "PROCESS_SCHEMAS",
    "validate_schema",
    "validate_composite_image_spec",
    "validate_vlm_recognition_spec",
    "validate_redaction_mapping_spec",
    "validate_rewritten_text_spec",
    "validate_payload_message_spec",
]


