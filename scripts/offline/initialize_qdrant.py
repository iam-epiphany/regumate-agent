"""Resume optimizers after snapshot recovery before accepting application traffic."""
from __future__ import annotations

import json
import os
import time
from urllib.parse import quote
from urllib.request import build_opener, ProxyHandler, Request


def main() -> None:
    base = os.environ.get('QDRANT_URL', 'http://qdrant:6333').rstrip('/')
    collection = os.environ.get('QDRANT_COLLECTION', 'regumate_chunks')
    endpoint = base + '/collections/' + quote(collection, safe='')
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with opener.open(endpoint, timeout=20) as response:
            info = json.load(response)['result']
        status = info['status']
        if status == 'grey':
            # Official recovery action; it does not alter vectors or payloads.
            request = Request(endpoint, data=b'{"optimizers_config":{}}',
                              headers={'Content-Type': 'application/json'}, method='PATCH')
            with opener.open(request, timeout=30) as response:
                json.load(response)
            print('Resumed pending collection optimizations.', flush=True)
        elif status in ('green', 'yellow') and info.get('points_count', 0) > 0:
            print(f'Knowledge collection is available ({status}, {info["points_count"]} points).', flush=True)
            return
        elif status == 'red':
            raise RuntimeError('Qdrant collection reports a failed state.')
        time.sleep(1)
    raise RuntimeError('Qdrant collection did not become available after snapshot recovery.')


if __name__ == '__main__':
    main()
