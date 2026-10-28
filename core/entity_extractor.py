"""Entity extraction and text-to-image prompt composition module."""

import json
import re
from typing import Dict, List, Any, Optional
from process_utils import extract_process_blocks, load_block_json, validate_schema
from .llm_client import llm_client
import prompts


class EntityExtractor:
    """Extracts visualization-friendly entities from text using an LLM."""
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entity candidates from text via LLM."""
        print("[INFO] Building entity extraction prompt...")
        prompt = prompts.get_prompt("entity_extraction", text=text)
        
        print("[INFO] Calling LLM API...")
        response = llm_client.call_text_model(prompt, "entity_extraction")
        print(f"[INFO] Received LLM response, length: {len(response)} characters")
        
        # Debug: save the raw LLM response
        import os
        os.makedirs("debug", exist_ok=True)
        with open("debug/llm_entity_response.txt", "w", encoding="utf-8") as f:
            f.write(f"Prompt:\n{prompt}\n\n")
            f.write(f"Response:\n{response}\n")
        
        # Parse the <process> block
        print("[INFO] Parsing process blocks...")
        blocks = extract_process_blocks(response)
        print(f"[INFO] Parsed {len(blocks)} process blocks")
        
        entity_blocks = [b for b in blocks if b.name == "EntityList"]
        
        if not entity_blocks:
            print(f"[ERROR] No EntityList block found!")
            print(f"[INFO] First 500 characters of LLM response:")
            print(f"{response[:500]}...")
            
            # Try to extract the JSON array directly
            import json
            try:
                # Find the JSON array
                json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
                if json_match:
                    print(f"[INFO] Found JSON array, attempting direct parse...")
                    entities_data = json.loads(json_match.group(0))
                    print(f"[SUCCESS] Direct JSON parse succeeded, extracted {len(entities_data)} entities")
                    return entities_data
            except Exception as e:
                print(f"[ERROR] Direct JSON parse also failed: {e}")
            
            raise ValueError("No EntityList found in LLM response")
        
        print(f"[INFO] Found EntityList block, loading JSON...")
        entities_data = load_block_json(entity_blocks[0])
        print(f"[SUCCESS] JSON parse succeeded, extracted {len(entities_data)} entities")
        
        # Validate schema
        print("[INFO] Validating entity schema...")
        for i, entity in enumerate(entities_data):
            errors = validate_schema(entity, "EntityList")
            if errors:
                print(f"[WARNING] Entity {i+1} schema validation failed: {errors}")
                # Do not raise an error, only warn
        
        return entities_data


class ImagePromptComposer:
    """Composes an image prompt by combining entities with LLM guidance."""
    
    def _add_distraction_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate and append distraction entities (strategy v3).
        
        Args:
            entities: Original entity list
            
        Returns:
            A new list with distraction entities merged
        """
        import config
        from process_utils import extract_process_blocks, load_block_json
        
        # Get the distraction entity configuration
        distraction_config = getattr(config, "DISTRACTION_ENTITY_CONFIG", {})
        if not distraction_config.get("enabled", True):
            return entities
        
        # Compute the number of distraction entities
        original_count = len(entities)
        mode = distraction_config.get("mode", "ratio")  # Default to ratio mode
        
        if mode == "fixed":
            # Fixed count mode: use fixed_count directly
            target_count = distraction_config.get("fixed_count", 3)
            print(f"[INFO] Distraction entity mode: fixed count ({target_count})")
        else:
            # Ratio mode: compute proportionally from the original entity count
            ratio = distraction_config.get("count_ratio", 0.3)
            min_count = distraction_config.get("min_count", 2)
            max_count = distraction_config.get("max_count", 5)
            target_count = max(min_count, min(max_count, int(original_count * ratio)))
            print(f"[INFO] Distraction entity mode: ratio mode ({ratio*100:.0f}%, range {min_count}-{max_count})")
        
        # Prepare the existing entity list (for the LLM to evaluate)
        existing_entities_str = "\n".join([f"- {e['name']} ({e.get('type', 'unknown')})" for e in entities])
        categories_str = ", ".join(distraction_config.get("categories", ["natural", "everyday", "abstract"]))
        
        # Call the LLM to generate distraction entities
        print(f"[INFO] Generating {target_count} distraction entities...")
        
        # Add a timestamp and random seed hint to force different results each time
        import time
        timestamp = int(time.time() * 1000) % 100000  # Use the last 5 digits of the millisecond timestamp as the random seed
        randomness_hint = f"\n\n**🎲 Random seed: {timestamp}** - Use this seed to ensure this generation is completely different from previous ones."
        
        prompt = prompts.get_prompt(
            "distraction_entity_generation",
            count=target_count,
            existing_entities=existing_entities_str,
            categories=categories_str
        ) + randomness_hint
        
        try:
            response = llm_client.call_text_model(prompt, "entity_extraction")
            
            # Parse the <process> block
            blocks = extract_process_blocks(response)
            distraction_blocks = [b for b in blocks if b.name == "DistractionEntities"]
            
            if not distraction_blocks:
                print(f"[WARNING] No DistractionEntities block found, attempting direct JSON parse...")
                # Try to extract the JSON array directly
                import json
                import re
                json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
                if json_match:
                    distraction_entities = json.loads(json_match.group(0))
                else:
                    print(f"[ERROR] Could not parse distraction entities, skipping")
                    return entities
            else:
                distraction_entities = load_block_json(distraction_blocks[0])
            
            print(f"[SUCCESS] Generated {len(distraction_entities)} distraction entities")
            for i, e in enumerate(distraction_entities[:3], 1):
                print(f"  {i}. {e.get('name', 'N/A')} ({e.get('type', 'N/A')})")
            if len(distraction_entities) > 3:
                print(f"  ... and {len(distraction_entities)-3} more")
            
            # Merge original entities with distraction entities
            return entities + distraction_entities
            
        except Exception as e:
            print(f"[ERROR] Distraction entity generation failed: {e}")
            return entities
    
    def compose_prompt(self, entities: List[Dict[str, Any]], narrative_style: str = "surreal_fusion") -> Dict[str, Any]:
        """Compose entities into a text-to-image prompt using an LLM.
        
        Retries ensure entity mapping completeness (target: 90%+ mapping rate).
        """
        import config
        
        # v3 strategy: generate distraction entities
        strategy = getattr(config, "PROMPT_STRATEGY", "v2")
        if strategy == "v3":
            entities = self._add_distraction_entities(entities)
            print(f"[INFO] v3 strategy: distraction entities added, total entities: {len(entities)}")
        
        entity_count = len(entities)
        min_required_entities = max(int(entity_count * 0.9), entity_count - 3)  # At least 90% or all minus 3
        
        max_retries = 3
        for attempt in range(max_retries):
            # Prepare input
            entities_json = json.dumps(entities, ensure_ascii=False, indent=2)
            
            # Select the prompt version
            # New combination: strategy x style; compatible with legacy PROMPT_VERSIONS
            strategy = getattr(config, "PROMPT_STRATEGY", None)
            style = getattr(config, "PROMPT_STYLE", None)
            prompt_name = None
            # Generate style directives; default to an empty string for compatibility
            style_directives = ""
            if strategy and style:
                normalized_style = str(style).strip().lower()
                if strategy == "v1" and normalized_style in ("realistic", "oil_painting"):
                    prompt_name = "image_composition_v1"
                elif strategy in ("v2", "v3"):  # v3 uses the same template as v2; the difference is at the entity layer
                    # The V3 template suits flat, cartoon styles (thick lines + flat colors + game style)
                    if normalized_style in ("cartoon", "comic", "pixel_art", "line_art", "isometric", "paper_cutout", "lowpoly",
                                           "minecraft", "voxel_art", "retro_8bit", "retro_16bit", "game_boy", "cel_shaded_3d"):
                        prompt_name = "image_composition_v3"
                    # The V2 template suits realistic, painting, technical drawing, and abstract styles
                    elif normalized_style in ("realistic", "sketch", "oil_painting", "watercolor", "blueprint", 
                                             "collage", "infrared", "xray", "pointillism", "glitch_art", "ascii_art", "ps1_style"):
                        prompt_name = "image_composition_v2"
                    else:
                        prompt_name = "image_composition_v2"
                else:
                    prompt_name = "image_composition_v2"

                # Inject finer-grained style hints (in English, to avoid affecting language rules)
                if normalized_style == "realistic":
                    style_directives = (
                        "**CRITICAL STYLE ENFORCEMENT - REALISTIC PHOTOGRAPHY**: "
                        "Generate photorealistic image with professional camera quality. "
                        "MANDATORY: accurate perspective and vanishing points; consistent natural lighting "
                        "(single light source with realistic shadows and highlights); "
                        "fine surface details (scratches, wear, reflections, subsurface scattering); "
                        "atmospheric depth cues (aerial perspective, focus falloff); "
                        "physically accurate materials (metal specularity, fabric weave, liquid meniscus); "
                        "NO cartoon outlines, NO flat shading, NO stylization. "
                        "Think: high-end product photography or documentary photography.\n"
                    )
                elif normalized_style == "sketch":
                    style_directives = (
                        "Add STYLE GUIDANCE: graphite pencil sketch look; monochrome/limited grayscale palette; "
                        "visible cross-hatching and contour lines; paper texture hints; high silhouette clarity; "
                        "no filled photo textures; keep lighting coherent with soft shading.\n"
                    )
                elif normalized_style == "cartoon":
                    style_directives = (
                        "**CRITICAL STYLE ENFORCEMENT - CLASSIC CARTOON**: "
                        "Generate in classic hand-drawn cartoon style (Disney/Warner Bros). "
                        "MANDATORY: thick black ink outlines (3-5px) around ALL shapes; "
                        "flat cel-shaded colors with NO gradients (use 2-3 discrete tones max per object); "
                        "exaggerated proportions and simplified anatomy; "
                        "white highlights as separate flat shapes (not gradients); "
                        "clean solid backgrounds; squash-and-stretch principles; "
                        "NO photorealistic textures, NO subtle shading, NO 3D rendering look. "
                        "Think: classic 2D animation cels.\n"
                    )
                elif normalized_style == "comic":
                    style_directives = (
                        "**CRITICAL STYLE ENFORCEMENT - COMIC BOOK ART**: "
                        "Generate in American comic book illustration style (Marvel/DC). "
                        "MANDATORY: bold black ink contour lines; "
                        "flat color fills with cel shading (hard-edged shadows); "
                        "dynamic composition with strong silhouettes; "
                        "Ben-Day dots or halftone patterns for midtones (optional); "
                        "high contrast lighting (noir-style drama); "
                        "NO speech bubbles, NO text, NO panel borders. "
                        "Think: printed comic panel ready for publication.\n"
                    )
                elif normalized_style == "oil_painting":
                    style_directives = (
                        "Add STYLE GUIDANCE: oil painting aesthetic; visible brushstrokes; impasto texture hints; "
                        "rich but controlled color harmony; soft edge transitions consistent with single light direction; "
                        "avoid photographic noise; no text overlays.\n"
                    )
                elif normalized_style == "watercolor":
                    style_directives = (
                        "Add STYLE GUIDANCE: watercolor wash and granulation; soft bleeding edges; limited harmonious palette; "
                        "paper texture hints; gentle value transitions; avoid hard photographic textures; no text overlays.\n"
                    )
                elif normalized_style == "pixel_art":
                    style_directives = (
                        "**CRITICAL STYLE ENFORCEMENT - PIXEL ART**: "
                        "Generate in pure pixel art style (16-bit game era). "
                        "MANDATORY: visible pixel grid (each pixel must be individually distinguishable); "
                        "ZERO anti-aliasing (crisp hard edges only); "
                        "limited color palette (max 64 colors); "
                        "dithering for gradients (checkerboard or ordered patterns); "
                        "manual pixel-by-pixel placement aesthetic; "
                        "isometric or side-view perspective preferred; "
                        "NO smooth curves, NO blur, NO gradients, NO modern rendering. "
                        "Think: Super Nintendo or Sega Genesis game sprites.\n"
                    )
                elif normalized_style == "blueprint":
                    style_directives = (
                        "Add STYLE GUIDANCE: blueprint aesthetic; cyan/indigo background with white/near-white technical linework; "
                        "orthographic/axonometric views; consistent line weights and construction marks; no labels or text allowed.\n"
                    )
                elif normalized_style == "line_art":
                    style_directives = (
                        "Add STYLE GUIDANCE: clean line art; pure black lines on white background; "
                        "minimal line weights (1-3px variation); no shading or fills; "
                        "vector-style crisp outlines; emphasize shape silhouettes and contours; "
                        "no gradients or textures; technical illustration precision.\n"
                    )
                elif normalized_style == "isometric":
                    style_directives = (
                        "Add STYLE GUIDANCE: isometric projection; 30-degree angle oblique view; "
                        "technical diagram aesthetic; clean geometric shapes; limited corporate color palette; "
                        "flat shading with no perspective distortion; grid-aligned placement; "
                        "emphasize spatial relationships through layering; no labels or annotations.\n"
                    )
                elif normalized_style == "collage":
                    style_directives = (
                        "Add STYLE GUIDANCE: mixed-media collage; disparate visual styles coexisting; "
                        "torn paper edges; overlapping layers with varied opacity; "
                        "intentional style clashes (photo-realistic fragments + flat graphics + textures); "
                        "disrupted spatial coherence; no unified lighting or perspective; "
                        "magazine cutout aesthetic; avoid text elements.\n"
                    )
                elif normalized_style == "infrared":
                    style_directives = (
                        "**CRITICAL STYLE ENFORCEMENT - THERMAL INFRARED CAMERA**: "
                        "Generate thermal imaging camera output (FLIR style). "
                        "MANDATORY: false-color heat map palette ONLY (cold=deep purple/blue → warm=yellow/orange → hot=red/white); "
                        "ZERO natural object colors (everything shown by temperature); "
                        "soft glowing edges with heat bloom halos; "
                        "high-contrast hot spots appear as bright white centers; "
                        "suppress ALL surface details and textures (only thermal silhouettes visible); "
                        "dark cold backgrounds; temperature gradients as smooth color transitions; "
                        "NO realistic colors, NO sharp edges, NO material textures. "
                        "Think: FLIR thermal camera forensic footage.\n"
                    )
                elif normalized_style == "xray":
                    style_directives = (
                        "Add STYLE GUIDANCE: X-ray radiography look; inverted luminance (bright = less dense); "
                        "semi-transparent overlapping forms; reveal internal structures; "
                        "monochrome or cyan-tinted palette; soft edge falloff; "
                        "bone/metal appear bright, soft tissues appear dim; "
                        "ethereal ghosting effect; no surface colors or photo textures.\n"
                    )
                elif normalized_style == "lowpoly":
                    style_directives = (
                        "Add STYLE GUIDANCE: low-polygon 3D render; visible triangular facets; "
                        "flat-shaded polygons; no smooth subdivision; "
                        "limited vertex count aesthetic; hard geometric edges; "
                        "solid matte colors per face; simple directional lighting; "
                        "origami-like angular forms; early-3D-game aesthetic.\n"
                    )
                elif normalized_style == "pointillism":
                    style_directives = (
                        "Add STYLE GUIDANCE: pointillist technique; composed entirely of small colored dots; "
                        "no continuous lines or fills; optical color mixing from distance; "
                        "visible dot grid pattern; impressionist palette; "
                        "form emerges from dot density and color juxtaposition; "
                        "soft silhouettes; avoid sharp edges or text.\n"
                    )
                elif normalized_style == "glitch_art":
                    style_directives = (
                        "Add STYLE GUIDANCE: digital glitch aesthetics; RGB channel separation/displacement; "
                        "horizontal scanline distortion; pixelation artifacts; "
                        "compression noise and blocking; color banding; "
                        "partial image corruption; datamoshing effects; "
                        "intentional visual errors; maintain recognizable base forms beneath glitches.\n"
                    )
                elif normalized_style == "paper_cutout":
                    style_directives = (
                        "Add STYLE GUIDANCE: layered paper cut-out art; multiple depth planes; "
                        "crisp cut edges with slight shadow gaps between layers; "
                        "flat matte card-stock textures; limited color palette per layer; "
                        "no gradients or photo realism; silhouette emphasis; "
                        "diorama-like depth; avoid text or fine details.\n"
                    )
                elif normalized_style == "ascii_art":
                    style_directives = (
                        "Add STYLE GUIDANCE: ASCII/Unicode character art; monospace text grid; "
                        "use character density to represent value (@ # + . space); "
                        "high contrast black-on-white or terminal green; "
                        "blocky pixelated appearance; no anti-aliasing; "
                        "emphasize recognizable silhouettes through character patterns.\n"
                    )
                elif normalized_style == "minecraft":
                    style_directives = (
                        "Add STYLE GUIDANCE: Minecraft voxel aesthetic; strict cubic blocks on 1-unit grid; "
                        "pixelated 16x16 block textures; no smooth curves or rounded edges; "
                        "orthogonal stacking; simple flat-shaded cubes; "
                        "limited material palette (stone/wood/metal/glass block types); "
                        "maintain cubic geometry; no anti-aliasing; blocky silhouettes only.\n"
                    )
                elif normalized_style == "voxel_art":
                    style_directives = (
                        "Add STYLE GUIDANCE: voxel art; 3D pixel grid aesthetic; cubic building blocks; "
                        "isometric or orthographic view; limited color palette per voxel; "
                        "flat-shaded cubes with no texture gradients; "
                        "stair-stepped edges; no smooth interpolation; "
                        "emphasize chunky geometric forms; Lego-like construction.\n"
                    )
                elif normalized_style == "retro_8bit":
                    style_directives = (
                        "Add STYLE GUIDANCE: 8-bit retro game aesthetic (NES/Famicom era); "
                        "extremely limited color palette (max 16-32 colors total); "
                        "chunky 8x8 or 16x16 sprite tiles; visible pixel grid; "
                        "no gradients or anti-aliasing; dithering for shading; "
                        "side-view or top-down perspective; flat backgrounds; "
                        "harsh color contrasts; black outlines around sprites.\n"
                    )
                elif normalized_style == "retro_16bit":
                    style_directives = (
                        "Add STYLE GUIDANCE: 16-bit retro game aesthetic (SNES/Genesis era); "
                        "richer but still limited color palette (256 colors); "
                        "detailed pixel sprites with dithered shading; "
                        "parallax layered backgrounds; mode-7 style pseudo-3D if applicable; "
                        "smooth pixel gradients; no anti-aliasing but more color depth; "
                        "vibrant saturated colors; detailed tile-based environments.\n"
                    )
                elif normalized_style == "game_boy":
                    style_directives = (
                        "Add STYLE GUIDANCE: Game Boy aesthetic; strict 4-color monochrome palette "
                        "(dark green, medium green, light green, off-white); "
                        "low resolution pixel art; extremely simple forms; "
                        "minimal shading with 2-3 value steps; high contrast; "
                        "tile-based composition; no gradients; pure pixel art clarity.\n"
                    )
                elif normalized_style == "ps1_style":
                    style_directives = (
                        "Add STYLE GUIDANCE: PlayStation 1 era 3D; low-polygon models with visible facets; "
                        "affine texture mapping with warping/distortion; "
                        "dithered gradients and textures; vertex jitter/wobble aesthetic; "
                        "compressed JPEG-like texture artifacts; low resolution (320x240 feel); "
                        "muted colors with color banding; hard shadows; nostalgic early-3D look.\n"
                    )
                elif normalized_style == "cel_shaded_3d":
                    style_directives = (
                        "**CRITICAL STYLE ENFORCEMENT - CEL-SHADED 3D (Zelda Wind Waker / Borderlands)**: "
                        "Generate 3D rendered scene with cartoon shading. "
                        "MANDATORY: thick black outline strokes around ALL 3D forms (toon shader outline); "
                        "posterized lighting with EXACTLY 2-3 discrete color bands per surface (NO smooth gradients); "
                        "hard-edged shadow boundaries (terminator line clearly visible); "
                        "vibrant saturated colors (anime-style palette); "
                        "simplified but volumetric 3D geometry (rounded edges, stylized proportions); "
                        "rim lighting as separate bright band; "
                        "NO photorealistic materials, NO smooth Phong/Blinn shading, NO raytracing. "
                        "Think: The Legend of Zelda Wind Waker or Genshin Impact rendering style.\n"
                    )
            else:
                version = getattr(config, "PROMPT_VERSIONS", {}).get("image_composition", "v2").lower()
                if version == "v3":
                    prompt_name = "image_composition_v3"
                elif version == "v1":
                    prompt_name = "image_composition_v1"
                else:
                    prompt_name = "image_composition_v2"
            prompt = prompts.get_prompt(
                prompt_name,
                entities_json=entities_json,
                entity_count=entity_count,
                style_directives=style_directives
            )
            
            response = llm_client.call_text_model(prompt, "narrative_composition")
            
            # Parse the <process> block
            blocks = extract_process_blocks(response)
            spec_blocks = [b for b in blocks if b.name == "ImagePrompt"]
            
            if not spec_blocks:
                if attempt < max_retries - 1:
                    print(f"[WARNING] Attempt {attempt+1}: no ImagePrompt found, retrying...")
                    continue
                raise ValueError("No ImagePrompt found in LLM response after retries")
            
            spec_data = load_block_json(spec_blocks[0])
            
            # Check the entity mapping rate
            mapped_count = len(spec_data.get('entities_mapped', []))
            mapping_rate = mapped_count / entity_count if entity_count > 0 else 0
            
            if mapped_count >= min_required_entities:
                print(f"[SUCCESS] Entity mapping rate: {mapped_count}/{entity_count} ({mapping_rate*100:.1f}%)")
                return spec_data
            else:
                print(f"[WARNING] Attempt {attempt+1}: mapping rate too low: {mapped_count}/{entity_count} ({mapping_rate*100:.1f}%), need >= {min_required_entities}")
                if attempt < max_retries - 1:
                    print(f"[INFO] Regenerating to improve mapping rate...")
                    continue
                else:
                    # Final attempt: return even if the requirement is not met
                    print(f"[WARNING] Maximum retries reached, returning current result")
                    return spec_data
        
        return spec_data


def generate_composite_image_spec(problem_id: str, entities: List[Dict[str, Any]], 
                                style: str = "diagram") -> Dict[str, Any]:
    """Generate a CompositeImageSpec."""
    
    # Select the top 3-6 high-scoring entities
    selected_entities = entities[:min(6, len(entities))]
    
    spec = {
        "problem_id": problem_id,
        "style": style,
        "layout": {
            "w": 1536,
            "h": 1024, 
            "theme": "neutral-technical"
        },
        "entities": [
            {
                "id": f"E{i+1}",
                "canonical": entity["canonical"],
                "hints": [
                    entity["name"],
                    f"Type: {entity['type']}",
                    f"Visualizability: {entity['visualizability']:.2f}"
                ]
            }
            for i, entity in enumerate(selected_entities)
        ],
        "contextual_neutrals": ["simple background", "neutral props"],
        "safety_constraints": [
            constraint for entity in selected_entities 
            if entity.get("safety_note") 
            for constraint in [entity["safety_note"]]
        ],
        "seed": 42
    }
    
    return spec


# Prompt templates
ENTITY_EXTRACTION_PROMPT = """Output only a JSON array wrapped in <process name="EntitySpecList">, with elements {{name, canonical, type, salience, visualizability, iconicity, safety_note}}. Do not output any other text.

{raw_text}"""

IMAGE_GENERATION_PROMPT = """Based on the following entities, generate a single cross-narrative image, and output JSON wrapped in <process name="CompositeImageSpec"> plus the corresponding text-to-image prompt.

Entity list: {entities}

Requirements:
1. Select 3-6 entities with high "visualizability x salience"
2. Create a new visual narrative; avoid directly reproducing the original context
3. Ensure image elements can be mapped back to entity categories
4. The output format must include CompositeImageSpec and an image_prompt field"""
