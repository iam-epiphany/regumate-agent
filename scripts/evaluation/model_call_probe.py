"""Optional evaluation-only transport counter; records no prompts or credentials.

Install before starting the frozen application. Each actual HTTP model attempt,
including a failed attempt, receives the application's current request ID.
"""
import json
from pathlib import Path
from time import time
import urllib.request


def install(destination: str) -> None:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = urllib.request.urlopen

    def counted(request, *args, **kwargs):
        model_request = '/chat/completions' in getattr(request, 'full_url', str(request))
        try:
            return original(request, *args, **kwargs)
        finally:
            if model_request:
                from backend.app.services.performance_metrics import _CURRENT_REQUEST_ID
                record = {'at': time(), 'request_id': _CURRENT_REQUEST_ID.get()}
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record) + '\n')

    urllib.request.urlopen = counted
