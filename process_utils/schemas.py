from typing import Any, Dict, List, Optional, Union



PROCESS_SCHEMAS = {
    "EntityList": {
        "note": "Entity extraction result (simplified)",
        "fields": [
            {"field": "name", "type": "str", "required": True, "desc": "Entity name"},
            {"field": "canonical", "type": "str", "required": True, "desc": "Canonical name"},
            {"field": "type", "type": "str", "required": True, "desc": "Entity type"}
        ]
    },
    "ProcessResult": {
        "note": "Generic processing result (flexible schema)",
        "fields": [
            {"field": "type", "type": "str", "required": True, "desc": "Result type"},
            {"field": "data", "type": "any", "required": True, "desc": "Result data"},
            {"field": "metadata", "type": "dict", "required": False, "desc": "Metadata"}
        ]
    }
}


def _get_nested_value(obj: Any, path: str) -> Any:
    """Get a value from a nested object, supporting dot-separated paths and array indices"""
    if not path:
        return obj
    
    parts = path.split('.')
    current = obj
    
    for part in parts:
        if part.endswith('[]'):
            # Array field; return the current value (should be an array)
            field_name = part[:-2]
            if isinstance(current, dict) and field_name in current:
                current = current[field_name]
            else:
                return None
        elif '[' in part and part.endswith(']'):
            # Array index access (not implemented yet; reserved for extension)
            return None
        else:
            # Regular field
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
    
    return current


def _validate_field(obj: Any, field_def: Dict[str, Any], path_prefix: str = "") -> List[str]:
    """Validate a single field and return a list of errors"""
    errors = []
    field_path = field_def["field"]
    field_type = field_def["type"]
    required = field_def.get("required", False)
    desc = field_def.get("desc", "")
    
    full_path = f"{path_prefix}.{field_path}" if path_prefix else field_path
    
    # Handle array fields
    if field_path.endswith('[]'):
        base_field = field_path[:-2]
        value = _get_nested_value(obj, base_field)
        
        if value is None:
            if required:
                errors.append(f"Missing required field: {full_path}")
            return errors
        
        if not isinstance(value, list):
            errors.append(f"Field {full_path} must be a list")
            return errors
        
        return errors
    
    # Handle array element fields
    if '[].' in field_path:
        # e.g. "entities[].id" -> validate the id field of each element in the array
        array_part, element_part = field_path.split('[].', 1)
        array_value = _get_nested_value(obj, array_part)
        
        if array_value is None:
            if required:
                errors.append(f"Missing array for field: {full_path}")
            return errors
        
        if not isinstance(array_value, list):
            errors.append(f"Expected list for {array_part}")
            return errors
        
        for i, item in enumerate(array_value):
            item_value = _get_nested_value(item, element_part)
            if item_value is None and required:
                errors.append(f"Missing required field: {array_part}[{i}].{element_part}")
            elif item_value is not None:
                # Type check
                if not _check_type(item_value, field_type):
                    errors.append(f"Field {array_part}[{i}].{element_part} has wrong type, expected {field_type}")
        
        return errors
    
    # Regular field
    value = _get_nested_value(obj, field_path)
    
    if value is None:
        if required:
            errors.append(f"Missing required field: {full_path}")
        return errors
    
    # Type check
    if not _check_type(value, field_type):
        errors.append(f"Field {full_path} has wrong type, expected {field_type}")
    
    return errors


def _check_type(value: Any, expected_type: str) -> bool:
    """Check whether the value's type matches the expected type"""
    if expected_type == "str":
        return isinstance(value, str)
    elif expected_type == "int":
        return isinstance(value, int)
    elif expected_type == "float":
        return isinstance(value, (int, float))
    elif expected_type == "bool":
        return isinstance(value, bool)
    elif expected_type == "list":
        return isinstance(value, list)
    elif expected_type == "dict":
        return isinstance(value, dict)
    elif expected_type == "any":
        return True
    else:
        return False


def validate_schema(data: Dict[str, Any], schema_name: str) -> List[str]:
    """Validate data against the specified schema and return a list of errors"""
    if schema_name not in PROCESS_SCHEMAS:
        return [f"Unknown schema: {schema_name}"]
    
    schema_def = PROCESS_SCHEMAS[schema_name]
    fields = schema_def.get("fields", [])
    
    errors = []
    for field_def in fields:
        field_errors = _validate_field(data, field_def)
        errors.extend(field_errors)
    
    return errors


# Compatibility functions (keep the existing interface)
def validate_composite_image_spec(spec: Dict[str, Any]) -> None:
    errors = validate_schema(spec, "CompositeImageSpec")
    if errors:
        raise ValueError(f"CompositeImageSpec validation failed: {'; '.join(errors)}")


def validate_vlm_recognition_spec(spec: Dict[str, Any]) -> None:
    errors = validate_schema(spec, "VLMRecognitionSpec")
    if errors:
        raise ValueError(f"VLMRecognitionSpec validation failed: {'; '.join(errors)}")


def validate_redaction_mapping_spec(spec: Dict[str, Any]) -> None:
    errors = validate_schema(spec, "RedactionMappingSpec")
    if errors:
        raise ValueError(f"RedactionMappingSpec validation failed: {'; '.join(errors)}")


def validate_rewritten_text_spec(spec: Dict[str, Any]) -> None:
    errors = validate_schema(spec, "RewrittenTextSpec")
    if errors:
        raise ValueError(f"RewrittenTextSpec validation failed: {'; '.join(errors)}")


def validate_payload_message_spec(spec: Dict[str, Any]) -> None:
    errors = validate_schema(spec, "PayloadMessageSpec")
    if errors:
        raise ValueError(f"PayloadMessageSpec validation failed: {'; '.join(errors)}")

