"""Copy points through the Qdrant API into Linux native storage and export a snapshot.

Run with writers quiesced and an empty, disposable target collection. This avoids
depending on host bind-mount snapshot copy semantics. No vectors are recomputed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.request import build_opener, ProxyHandler
from urllib.parse import quote

from qdrant_client import QdrantClient, models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--collection', default='regumate_chunks')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = QdrantClient(url=args.source, timeout=180)
    target = QdrantClient(url=args.target, timeout=180)
    name = args.collection
    if target.collection_exists(name):
        raise RuntimeError('Target collection must be absent; existing data will not be overwritten.')
    info = source.get_collection(name)
    params = info.config.params
    target.create_collection(
        name, vectors_config=params.vectors, sparse_vectors_config=params.sparse_vectors,
        on_disk_payload=params.on_disk_payload, shard_number=params.shard_number,
        hnsw_config=info.config.hnsw_config.model_dump(exclude_none=True), quantization_config=info.config.quantization_config,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    offset = None
    count = 0
    samples = []
    started = time.monotonic()
    while True:
        records, offset = source.scroll(name, limit=128, offset=offset, with_payload=True, with_vectors=True)
        if records:
            target.upsert(name, points=[models.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records], wait=True)
            count += len(records)
            if len(samples) < 20:
                samples.append(records[0])
        if count % 2048 == 0 or offset is None:
            print(f'Portable copy: {count}/{info.points_count} points', flush=True)
        if offset is None:
            break
    for field, schema in info.payload_schema.items():
        target.create_payload_index(name, field, field_schema=schema.params or schema.data_type, wait=True)
    target.update_collection(name, optimizers_config=info.config.optimizer_config.model_dump(exclude_none=True))
    exact = target.count(name, exact=True).count
    if exact != count or exact != source.count(name, exact=True).count:
        raise RuntimeError('Point count changed during export or copy is incomplete.')
    copied = {str(r.id): r for r in target.retrieve(name, ids=[r.id for r in samples], with_payload=True, with_vectors=True)}
    for item in samples:
        match = copied[str(item.id)]
        if item.vector != match.vector or item.payload != match.payload:
            raise RuntimeError('Copied vector/payload sample differs from source.')
    snapshot = target.create_snapshot(name, wait=True)
    opener = build_opener(ProxyHandler({}))
    url = args.target.rstrip('/') + '/collections/' + quote(name, safe='') + '/snapshots/' + quote(snapshot.name, safe='')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix('.pending')
    with opener.open(url, timeout=600) as response, pending.open('wb') as stream:
        while block := response.read(8 * 1024 * 1024):
            stream.write(block)
    if pending.stat().st_size != snapshot.size:
        raise RuntimeError('Snapshot download is incomplete.')
    pending.replace(args.output)
    print(json.dumps({'points': exact, 'verified_vector_payload_samples': len(samples),
                      'seconds': round(time.monotonic() - started, 2), 'snapshot_bytes': snapshot.size}), flush=True)


if __name__ == '__main__':
    main()
