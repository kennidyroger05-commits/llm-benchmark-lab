"""Small, sanitized failure records for Phase 2 RAW artifacts (never prompts)."""
import os
import re


def sanitized_message(message):
    """Keep useful API diagnostics, excluding credentials, paths and env dumps.

    Inspect only values of credential-like environment keys; never copy the
    environment into a diagnostic. Apply redaction before limiting its length.
    """
    text = str(message)
    if re.search(r'\benviron\s*\(|\bos\.environ\b', text):
        return '[redacted environment diagnostic]'
    for key, value in os.environ.items():
        if value and re.search(r'token|secret|password|api.?key|credential|auth', key, re.I):
            text = text.replace(value, '[redacted]')
    text = re.sub(r'(?i)\b(?:proxy-)?authorization[\"\']?\s*[:=]\s*[\"\']?'
                  r'(?:(?:bearer|basic|token)\s+)?[^\s,;\"\']+', '[redacted authorization]', text)
    text = re.sub(r'https?://\S+', '[redacted URL]', text)
    text = re.sub(r'(?i)\bBearer\s+\S+', 'Bearer [redacted]', text)
    text = re.sub(r'\b(?:hf_|ghp_|github_pat_)[A-Za-z0-9_]+', '[redacted]', text)
    # Credential labels end in TOKEN/SECRET/etc. Do not erase useful API names
    # such as max_new_tokens, token_type_ids or tokenizer in the actual error.
    text = re.sub(r'(?i)\b(?:[\w-]+[_-])?(?:token|secret|password|api_key|credential)\b'
                  r'(?:\s*[=:]\s*[\"\']?[^\s,;\"\']+)?', '[redacted]', text)
    text = re.sub(r'(?<!\w)(?:[A-Za-z]:[\\/]|/)[^\s\"\'<>]*', '[redacted path]', text)
    return ' '.join(text.split())[:512]


def failure_details(error, *, stage, task_id=None, turn_index=None):
    """Project an exception, without traceback/args, or enrich a provider record."""
    original = getattr(error, 'diagnostic', None)
    if isinstance(original, dict):
        kind = original.get('exception_type', type(error).__name__)
        message = original.get('exception_message', '')
        stage = original.get('stage', stage)
    else:
        kind, message = type(error).__name__, str(error)
    # Exception classes/stages are identifiers, not arbitrary exception payloads.
    kind = kind if isinstance(kind, str) and re.fullmatch(r'[A-Za-z_]\w{0,79}', kind) else 'Exception'
    stage = stage if isinstance(stage, str) and re.fullmatch(r'[a-z_]{1,64}', stage) else 'provider'
    return {'exception_type': kind, 'exception_message': sanitized_message(message),
            'stage': stage, 'task_id': task_id, 'turn_index': turn_index}
