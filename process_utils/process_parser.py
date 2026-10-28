import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


PROCESS_BLOCK_PATTERN = re.compile(
    r"<process\b([^>]*)>([\s\S]*?)</process>", re.IGNORECASE
)
ATTR_PATTERN = re.compile(r"(\w+)\s*=\s*\"([^\"]*)\"")

# XML tag parsing patterns
ORIGIN_PATTERN = re.compile(r"<ORIGIN>(.*?)</ORIGIN>", re.DOTALL)
GLOBAL_PATTERN = re.compile(r"<GLOBAL>(.*?)</GLOBAL>", re.DOTALL)
LOCAL_PATTERN = re.compile(r"<LOCAL_(\d+)>(.*?)</LOCAL_\1>", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<ANSWER_(\d+)>(.*?)</ANSWER_\1>", re.DOTALL)


@dataclass
class ProcessBlock:
    raw: str
    attrs: Dict[str, str]
    content: str

    @property
    def name(self) -> Optional[str]:
        return self.attrs.get("name")


@dataclass
class LocalAnswerPair:
    """Pairing of LOCAL and ANSWER"""
    index: int
    local_content: str
    answer_content: str


@dataclass
class StructuredQuery:
    """Structured query parsing result"""
    origin: Optional[str] = None
    global_content: Optional[str] = None
    local_answer_pairs: List[LocalAnswerPair] = None
    raw_text: str = ""
    
    def __post_init__(self):
        if self.local_answer_pairs is None:
            self.local_answer_pairs = []


def extract_process_blocks(text: str) -> List[ProcessBlock]:
    """Extract process blocks, with fault-tolerant handling"""
    blocks: List[ProcessBlock] = []
    
    # Try standard parsing first
    for m in PROCESS_BLOCK_PATTERN.finditer(text):
        attr_str = m.group(1) or ""
        content = (m.group(2) or "").strip()
        attrs = dict(ATTR_PATTERN.findall(attr_str))
        blocks.append(ProcessBlock(raw=m.group(0), attrs=attrs, content=content))
    
    # If no blocks are found, try fault-tolerant parsing
    if not blocks:
        # Try to fix common formatting errors
        # Error 1: <process name="Entity\<process name="EntityList"\>
        # Fix to: <process name="EntityList">
        fixed_text = re.sub(
            r'<process\s+name="[^"]*\\<process\s+name="([^"]+)"\\>',
            r'<process name="\1">',
            text
        )
        
        # Error 2: Missing closing tag; try to add it
        if '<process' in fixed_text and '</process>' not in fixed_text:
            fixed_text += '\n</process>'
        
        # Try parsing again
        for m in PROCESS_BLOCK_PATTERN.finditer(fixed_text):
            attr_str = m.group(1) or ""
            content = (m.group(2) or "").strip()
            attrs = dict(ATTR_PATTERN.findall(attr_str))
            blocks.append(ProcessBlock(raw=m.group(0), attrs=attrs, content=content))
    
    return blocks


def get_blocks_by_name(blocks: List[ProcessBlock], name: str) -> List[ProcessBlock]:
    return [b for b in blocks if b.attrs.get("name") == name]


def load_block_json(block: ProcessBlock):
    """Load JSON data from a ProcessBlock"""
    try:
        return json.loads(block.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in <process name={block.attrs.get('name')}>: {e}")


def parse_structured_query(text: str) -> StructuredQuery:
    """Parse structured query text
    
    Supported format:
    <ORIGIN>original question</ORIGIN>
    <GLOBAL>
        <LOCAL_1>question 1</LOCAL_1><ANSWER_1>answer 1</ANSWER_1>
        <LOCAL_2>question 2</LOCAL_2><ANSWER_2>answer 2</ANSWER_2>
        ...
        global question part
    </GLOBAL>
    """
    result = StructuredQuery(raw_text=text)
    
    # Extract ORIGIN
    origin_match = ORIGIN_PATTERN.search(text)
    if origin_match:
        result.origin = origin_match.group(1).strip()
    
    # Extract GLOBAL content
    global_match = GLOBAL_PATTERN.search(text)
    if global_match:
        global_content = global_match.group(1).strip()
        
        # Extract LOCAL-ANSWER pairs from GLOBAL
        local_matches = LOCAL_PATTERN.findall(global_content)
        answer_matches = ANSWER_PATTERN.findall(global_content)
        
        # Build LOCAL-ANSWER pairs
        local_dict = {int(idx): content.strip() for idx, content in local_matches}
        answer_dict = {int(idx): content.strip() for idx, content in answer_matches}
        
        # Create the pair list
        for idx in sorted(set(local_dict.keys()) | set(answer_dict.keys())):
            if idx in local_dict and idx in answer_dict:
                pair = LocalAnswerPair(
                    index=idx,
                    local_content=local_dict[idx],
                    answer_content=answer_dict[idx]
                )
                result.local_answer_pairs.append(pair)
        
        # Remove LOCAL and ANSWER tags, keeping the remaining global content
        clean_global = global_content
        for pattern in [LOCAL_PATTERN, ANSWER_PATTERN]:
            clean_global = pattern.sub('', clean_global)
        
        result.global_content = clean_global.strip()
    
    return result


def reconstruct_complex_text(structured_query: StructuredQuery) -> str:
    """Reconstruct complex text from a structured query (for entity extraction)"""
    parts = []
    
    # Add all LOCAL content
    for pair in structured_query.local_answer_pairs:
        parts.append(pair.local_content)
    
    # Add global content
    if structured_query.global_content:
        parts.append(structured_query.global_content)
    
    return " ".join(parts)


def get_entity_mapping(structured_query: StructuredQuery) -> Dict[str, str]:
    """Get the entity mapping (LOCAL question -> ANSWER answer)"""
    mapping = {}
    for pair in structured_query.local_answer_pairs:
        # Simplified mapping: use the answer directly as the entity
        mapping[pair.answer_content] = pair.answer_content
    return mapping


def validate_structured_query(structured_query: StructuredQuery) -> Tuple[bool, List[str]]:
    """Validate the integrity of the structured query"""
    errors = []
    
    if not structured_query.origin:
        errors.append("Missing <ORIGIN> tag")
    
    if not structured_query.local_answer_pairs:
        errors.append("Missing LOCAL-ANSWER pairs")
    
    # Check LOCAL-ANSWER pair completeness
    for pair in structured_query.local_answer_pairs:
        if not pair.local_content:
            errors.append(f"LOCAL_{pair.index} content is empty")
        if not pair.answer_content:
            errors.append(f"ANSWER_{pair.index} content is empty")
    
    if not structured_query.global_content:
        errors.append("Missing global question content")
    
    return len(errors) == 0, errors


