# -*- coding: utf-8 -*-

"""
Entity migration and image anchoring module.
Pipeline: entity masking → VLM localization → position filling → LLM polishing.
"""

from typing import List, Dict, Any
from core.llm_client import llm_client
import base64
from json_repair import repair_json
import logging
import json
import re
import prompts
import config

logger = logging.getLogger(__name__)


def _lang_suffix() -> str:
    """Current output language (zh / en), determined by config.OUTPUT_LANGUAGE."""
    try:
        lang = getattr(config, "OUTPUT_LANGUAGE", "zh").lower()
    except Exception:
        lang = "zh"
    return "en" if lang.startswith("en") else "zh"


def mask_entities_in_text(complex_text: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stage 1: Mask entities in the text using simple rules.
    
    Args:
        complex_text: Original complex text
        entities: Entity list (from stage 1 extraction)
    
    Returns:
        {
            'masked_text': str,  # text after masking
            'mask_mappings': List[Dict],  # mask mapping relationships
        }
    """
    import re
    
    masked_text = complex_text
    mask_mappings = []
    
    # Sort entities by name length in descending order, matching longer entities first (to avoid short entities being contained in long entities)
    sorted_entities = sorted(entities, key=lambda x: len(x['name']), reverse=True)
    
    mask_counter = 1
    entity_to_mask = {}  # Record the entity-to-mask_id mapping so the same entity always uses the same mask
    
    for entity in sorted_entities:
        entity_name = entity['name']
        
        # If this entity has already been masked, skip it
        if entity_name in entity_to_mask:
            continue
        
        # Check whether the text contains this entity
        if entity_name in masked_text:
            mask_id = f"MASK_{mask_counter}"
            
            # Replace all occurrences of this entity
            masked_text = masked_text.replace(entity_name, f"[{mask_id}]")
            
            # Record the mapping relationship
            mask_mappings.append({
                'mask_id': mask_id,
                'original_entity': entity_name,
                'canonical': entity.get('canonical', entity_name)
            })
            
            entity_to_mask[entity_name] = mask_id
            mask_counter += 1
    
    return {
        'masked_text': masked_text,
        'mask_mappings': mask_mappings
    }


def vlm_locate_elements_in_image(image_path: str, entities: List[Dict[str, Any]], 
                                 max_retries: int = 3) -> Dict[str, Any]:
    """Stage 2: Use VLM to locate entity positions in the image (with retries).
    
    Args:
        image_path: Path to the generated image
        entities: Entity list (from stage 1; includes name, canonical, etc.)
        max_retries: Maximum retries when JSON parsing fails
    
    Returns:
        {
            'element_locations': List[Dict],  # element location descriptions
        }
    """
    # Read the image as base64
    import base64
    image_b64 = None
    try:
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        image_b64 = None

    # Build the list of entities to locate
    entities_list = '\n'.join([
        f"- {entity['name']}"
        for entity in entities
    ])
    
    # Read the lexicon from config (generic spatial terms; banned words default to empty to preserve generalization)
    try:
        import config
        allowed_list = config.LOCATION_LEXICON.get("allowed", [])
        banned_list = config.LOCATION_LEXICON.get("banned", [])
    except Exception:
        # No longer maintain the lexicon in code; default to no lexicon constraint (preserve generalization) and only perform leakage sanitization
        allowed_list = []
        banned_list = []

    allowed_vocab = ", ".join(allowed_list) if allowed_list else "(No hard restriction; choose appropriate spatial terms based on the image)"
    try:
        import config
        neutral_heads = getattr(config, "NEUTRAL_LOCATION_HEADS", [])
    except Exception:
        neutral_heads = []
    banned_terms = ", ".join(banned_list)

    locate_key = f"vlm_locate_elements_{_lang_suffix()}"
    prompt = prompts.get_prompt(
        locate_key,
        entities_list=entities_list,
        entity_count=len(entities),
        allowed_vocab=allowed_vocab,
        neutral_heads=", ".join(neutral_heads) if neutral_heads else "device/structure/element/unit",
    )
    
    # Retry loop: VLM call + JSON parsing
    import json
    import re
    import os
    result = {"element_locations": []}
    
    # Get the task config to pass max_tokens explicitly
    try:
        import config
        vlm_task_config = config.LLM_TASKS.get("vlm_recognition", {})
        vlm_max_tokens = vlm_task_config.get("max_tokens", 2048)  # Default to 2048 to accommodate more entities
    except Exception:
        vlm_max_tokens = 2048
    
    for attempt in range(max_retries):
        try:
            # Locate elements via the VLM or text model
            
            if image_b64:
                response = llm_client.call_vision_model(
                    prompt, image_b64, 
                    task_type="vlm_recognition",
                    max_tokens=vlm_max_tokens  # Pass explicitly
                )
            else:
                response = llm_client.call_text_model(prompt, "vlm_element_location")
            
# Debug: save raw VLM response
            os.makedirs("debug", exist_ok=True)
            with open(f"debug/vlm_location_response_attempt{attempt+1}.txt", "w", encoding="utf-8") as f:
                f.write(f"Attempt {attempt+1}/{max_retries}\n")
                f.write(f"Prompt:\n{prompt}\n\n{'='*60}\n\nResponse:\n{response}\n")
            
# Extract JSON code block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
    # If no code block, use response directly
                json_str = response.strip()
            
# Parse JSON (with json_repair for robustness)
            if repair_json:
# Use json_repair to fix potentially broken JSON
                repaired_str = repair_json(json_str)
                result = json.loads(repaired_str)
            else:
                result = json.loads(json_str)

            # Tolerance: some models return an array directly instead of a {"element_locations": [...]} object
            if isinstance(result, list):
                # Normalize the array response
                result = {"element_locations": result}
            elif isinstance(result, dict) and "element_locations" not in result:
                # Tolerance: promote the only list field containing original_entity
                for key, value in result.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict) \
                            and "original_entity" in value[0]:
                        # Promoted list field to element_locations
                        result = {"element_locations": value}
                        break

            # Parsing succeeded with results, break out of the retry loop
            if result.get("element_locations"):
                print(f"[SUCCESS] VLM localization succeeded on attempt {attempt+1}")
                break
            else:
                print(f"[WARNING] VLM returned empty list, retrying")
                if attempt < max_retries - 1:
                    continue
                    
        except json.JSONDecodeError as e:
            print(f"[ERROR] VLM localization JSON parse failed on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying (attempt {attempt+2})...")
                continue
            else:
                print(f"[ERROR] Max retries reached, returning empty result")
                result = {"element_locations": []}
        except Exception as e:
            print(f"[ERROR] VLM localization failed on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying (attempt {attempt+2})...")
                continue
            else:
                print(f"[ERROR] Max retries reached, returning empty result")
                result = {"element_locations": []}


    def llm_neutralize_location_phrases(phrases, heads):
        spec = {
            "phrases": phrases,
            "allowed_heads": heads or ["device", "structure", "element", "unit"]
        }
        payload = json.dumps(spec, ensure_ascii=False, indent=2)
        neutralize_prompt = prompts.get_prompt(
            f"vlm_neutralize_{_lang_suffix()}", payload=payload
        )
        resp = llm_client.call_text_model(neutralize_prompt, "text_rewriting")
        m = re.search(r'<process name="Neutralized">([\s\S]*?)</process>', resp)
        
        # Extract the JSON string
        json_str = m.group(1) if m else resp
        
        # Use json_repair for tolerant parsing
        from json_repair import repair_json
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Neutralization JSON parse failed, trying json_repair")
            data = repair_json(json_str)
        
        if isinstance(data, list):
            data = {"phrases": data}
        
        return data.get("phrases", [])

    if isinstance(result, dict) and isinstance(result.get("element_locations"), list):
        raw_phrases = [loc.get("location_phrase", "") for loc in result["element_locations"]]
        
        # Decide whether to enable neutralization based on config
        import config
        neutralization_config = getattr(config, "LOCATION_NEUTRALIZATION", {"enabled": False})
        
        if neutralization_config.get("enabled", False):
            # Enable neutralization: convert the VLM's concrete descriptions into abstract "location word + neutral noun"
            print(f"[INFO] Enabling neutralization mode - normalizing VLM location descriptions")
            neutralized_phrases = llm_neutralize_location_phrases(raw_phrases, neutral_heads)
            
            for i, loc in enumerate(result["element_locations"]):
                # Use the neutralized phrase
                if i < len(neutralized_phrases) and neutralized_phrases[i]:
                    original = raw_phrases[i] if i < len(raw_phrases) else ""
                    loc["location_phrase"] = neutralized_phrases[i]
                    print(f"[INFO] Neutralized: '{original}' → '{neutralized_phrases[i]}'")
                elif not loc.get("location_phrase"):
                    # The location phrase is empty; use the entity name as fallback
                    loc["location_phrase"] = loc.get("original_entity", "")
        else:
            # Disable neutralization: use the VLM's original location description directly
            # Keep the original location phrase
            for i, loc in enumerate(result["element_locations"]):
                if not loc.get("location_phrase"):
                    # The location phrase is empty; use the entity name as fallback
                    loc["location_phrase"] = loc.get("original_entity", "")
    
    return result


def fill_masks_with_locations(masked_text: str, mask_mappings: List[Dict], element_locations: List[Dict]) -> str:
    """Stage 3: Fill location descriptions into the mask placeholders.
    
    Args:
        masked_text: text after masking
        mask_mappings: mask mapping relationships
        element_locations: element location descriptions
    
    Returns:
        text after filling
    """
    # Build the entity-to-location mapping
    entity_to_location = {}
    for loc in element_locations:
        entity_to_location[loc['original_entity']] = loc['location_phrase']
    
    # Replace masks (keep boundary symbols for later polishing recognition)
    filled_text = masked_text
    for mapping in mask_mappings:
        mask_id = mapping['mask_id']
        original_entity = mapping['original_entity']
        
        # Find the corresponding location description
        location_phrase = entity_to_location.get(original_entity, entity_to_location.get(mapping.get('canonical', ''), ''))
        
        # If no usable location phrase exists, fall back to the original entity name (no generic fallback)
        replacement_core = location_phrase if location_phrase else original_entity
        replacement = f"[{replacement_core}]"
        filled_text = filled_text.replace(f"[{mask_id}]", replacement)
    
    return filled_text


def polish_text_coherence(filled_text: str, complex_text: str, image_path: str) -> Dict[str, Any]:
    """Stage 4: Use a multimodal model to polish against the image, ensuring semantic coherence and strong image binding.
    
    Args:
        filled_text: text after filling in location descriptions
        complex_text: original complex text (to ensure unreplaced parts stay faithful to the original)
    
    Returns:
        {
            'polished_text': str,  # polished text
            'coherence_score': float,  # Coherence score
        }
    """
    # Read the image as base64
    image_b64 = None
    try:
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        image_b64 = None

    prompt = prompts.get_prompt(
        f"vlm_polish_{_lang_suffix()}",
        original_text=filled_text,
        filled_text=filled_text,
    )
    
    # Select the polishing model based on config
    try:
        import config
        use_vlm = (getattr(config, "POLISHING_MODE", "vlm").lower() == "vlm")
    except Exception:
        use_vlm = True

    if use_vlm and image_b64:
        response = llm_client.call_vision_model(prompt, image_b64, task_type="vlm_text_polishing")
    else:
        response = llm_client.call_text_model(prompt, "text_polishing")
    
    # Validate whether the response is valid
    if not response or not isinstance(response, str):
        # Invalid response
        raise Exception(f"Text polishing failed: model returned an empty response or invalid type (type: {type(response)})")
    
    # Parse the JSON response
    import json
    import re
    from json_repair import repair_json
    
    # Extract the JSON code block
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # If there is no code block, use the response directly
        json_str = response.strip()
    
    # Check whether the JSON string is empty
    if not json_str:
        # Empty JSON string
        raise Exception(f"Text polishing failed: could not extract valid JSON from response\nOriginal response: {response[:500]}")
    
    # Use json_repair for tolerant parsing
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        # JSON parse failed; attempt repair
        try:
            result = repair_json(json_str)
        except Exception as repair_error:
            # Repair failed
            # Raise with context
            raise Exception(f"Text polishing failed: JSON parse error\nOriginal response: {response[:500]}\nJSON string: {json_str}")
    
    return result


def migrate_entities_to_image(
    complex_text: str,
    entities: List[Dict[str, Any]],
    image_path: str
) -> Dict[str, Any]:
    """Complete end-to-end entity migration pipeline.
    
    Args:
        complex_text: original complex text
        entities: entity list (extracted from stage 1)
        image_path: path to the generated image
    
    Returns:
        {
            'masked_text': str,
            'filled_text': str,
            'polished_text': str,
            'mask_mappings': List[Dict],
            'element_locations': List[Dict],
            'coherence_score': float,
            'pipeline_stages': Dict  # detailed information for each stage
        }
    """
    print("Entity migration pipeline starting...")
    print("=" * 60)
    
    # Stage 1: Mask entities
    print("Stage 1: Masking entities with LLM")
    print("-" * 40)
    import re as _rm
    mask_result = None
    for _m_try in range(3):
        mask_result = mask_entities_in_text(complex_text, entities)
        _left = [e["name"] for e in entities if e.get("name")
                 and _rm.search(r"\b" + _rm.escape(e["name"]) + r"\b", mask_result["masked_text"], _rm.I)]
        if not _left:
            break
        print(f"[GUARD] masking attempt {_m_try + 1} left entity names in text: {_left}")
    print(f"[SUCCESS] Masking complete, processed {len(mask_result['mask_mappings'])} entities")
    print(f"Masked text preview: {mask_result['masked_text'][:150]}...")
    
    # Stage 2: VLM locates elements
    # Locate elements in the image
    print("-" * 40)
    location_result = vlm_locate_elements_in_image(
        image_path,
        entities
    )
    # Location descriptions obtained
    for loc in location_result['element_locations'][:3]:
        print(f"  - {loc['original_entity']} → {loc['location_phrase']}")
    
    # Stage 3: Fill location descriptions
    print("-" * 40)
    filled_text = fill_masks_with_locations(
        mask_result['masked_text'],
        mask_result['mask_mappings'],
        location_result['element_locations']
    )
    # Fill masks with locations
    
    # Stage 4: VLM polishing (against the image)
    print("-" * 40)
    import re as _re
    _terms = set()
    for _e in entities:
        for _v in (_e.get("name", ""), _e.get("canonical", "")):
            if _v:
                _terms.add(_v.lower())
                _terms.update(w for w in _re.findall(r"[a-z]+", _v.lower()) if len(w) >= 4)

    def _leaks(text):
        low_f, low_t = filled_text.lower(), text.lower()
        out = []
        for t in _terms:
            pat = r"\b" + _re.escape(t) + r"(?:s|es)?\b"
            if len(_re.findall(pat, low_t)) > len(_re.findall(pat, low_f)):
                out.append(t)
        return out

    polish_result = None
    for _attempt in range(3):
        _cand = polish_text_coherence(filled_text, complex_text, image_path)
        _bad = _leaks(_cand["polished_text"])
        if not _bad:
            polish_result = _cand
            break
        print(f"[GUARD] polish attempt {_attempt + 1} reintroduced entity terms: {_bad}")
    if polish_result is None:
        print("[GUARD] all attempts leaked entity terms; falling back to the unpolished filled text")
        def _neutral(m):
            art = m.group(1) or ""
            head = "The" if art[:1].isupper() else "the"
            return f"{head} item {m.group(2)}"
        polish_result = {
            "polished_text": _re.sub(r"(\b(?:[Aa]n?|[Tt]he)\s+)?\[([^\[\]]+)\]", _neutral, filled_text),
            "coherence_score": 0.0,
        }
        polish_result["polished_text"] = _re.sub(
            r"\b(?:a|an|the)\s+(?:the\s+)?item\b", "the item", polish_result["polished_text"], flags=_re.I
        )
        polish_result["polished_text"] = _re.sub(r"^\s*the item\b", "The item", polish_result["polished_text"])
    # Polished text and coherence score obtained
    
    # Stage 5: LLM semantic correction (second polishing)
    print("-" * 40)
    
    import prompts
    import config
    
    # Select the output language based on config
    output_lang = getattr(config, "OUTPUT_LANGUAGE", "en").lower()
    prompt_key = f"llm_semantic_polish_{output_lang}" if output_lang in ["en", "zh"] else "llm_semantic_polish_en"
    
    semantic_polish_prompt = prompts.get_prompt(
        prompt_key,
        original_text=filled_text,
        vlm_polished_text=polish_result['polished_text']
    )
    
    llm_polished_response = llm_client.call_text_model(semantic_polish_prompt, "text_rewriting")
    
    # Clean up possible markdown markers
    llm_polished_text = llm_polished_response.replace('```', '').strip()
    
    # Remove all newlines so the text becomes a single paragraph (hide structure)
    # First replace multiple newlines (\n\n) with a single space, then replace single newlines with spaces
    llm_polished_text = llm_polished_text.replace('\n\n', ' ').replace('\n', ' ')
    # Clean up extra consecutive spaces
    import re
    llm_polished_text = re.sub(r'\s+', ' ', llm_polished_text).strip()
    _bad5 = _leaks(llm_polished_text)
    if _bad5:
        print(f"[GUARD] semantic polish reintroduced entity terms: {_bad5}; keeping the stage-4 text")
        llm_polished_text = re.sub(r'\s+', ' ', polish_result['polished_text'].replace('\n', ' ')).strip()
    
    print("\n" + "=" * 60)
    # Pipeline finished
    
    return {
        'masked_text': mask_result['masked_text'],
        'filled_text': filled_text,
        'vlm_polished_text': polish_result['polished_text'],  # VLM polishing result
        'polished_text': llm_polished_text,  # Final LLM correction result
        'mask_mappings': mask_result['mask_mappings'],
        'element_locations': location_result['element_locations'],
        'coherence_score': polish_result['coherence_score'],
        'pipeline_stages': {
            'stage1_mask': mask_result,
            'stage2_location': location_result,
            'stage3_filled': filled_text,
            'stage4_vlm_polish': polish_result,
            'stage5_llm_semantic_polish': {'polished_text': llm_polished_text}
        }
    }

