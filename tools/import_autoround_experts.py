"""Import pinned Flash Next AutoRound INT4 into an isolated inferzero expert file.

No model code is executed. All source tensor files are SHA-256 checked; every
code/scale is round-trip checked before publishing the completed expert file.
Usage: python import_autoround_experts.py SOURCE_DIRECTORY OUTPUT.experts
"""
import argparse
import hashlib
import json
import mmap
import os
import struct
import time
from pathlib import Path
import numpy as np

AR4 = 0x41523447
REVISION = '4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--wait-for-downloads', action='store_true')
    parser.add_argument('--experimental-format', action='store_true', help='Acknowledge private AR4_128 format; requires matching experimental runtime')
    args = parser.parse_args()
    if not args.experimental_format:
        parser.error('private experimental AR4_128 format: explicitly pass --experimental-format')
    if args.output.exists():
        raise ValueError('refusing to overwrite an existing expert file')
    info = json.loads((args.source / 'model-info.json').read_text(encoding='utf-8'))
    config = json.loads((args.source / 'config.json').read_text(encoding='utf-8'))
    weight_map = json.loads((args.source / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    files = {item['rfilename']: item for item in info['siblings']}
    quant = config['quantization_config']
    if info['id'] != 'Intel/Qwen3.8-Flash-Next-W4A16-AutoRound' or info['sha'] != REVISION:
        raise ValueError('unexpected checkpoint identity/revision')
    if quant['bits'] != 4 or quant['group_size'] != 128 or quant['sym'] is not True:
        raise ValueError('expected symmetric INT4/group-128')
    if quant['packing_format'] != 'auto_round:auto_gptq':
        raise ValueError('unsupported source packing convention')
    text = config['text_config']
    if (text['hidden_size'], text['moe_intermediate_size'], text['num_hidden_layers'],
            text['num_experts'], text['num_experts_per_tok'], text['max_position_embeddings']) != (2560, 640, 48, 512, 10, 262144):
        raise ValueError('unexpected source architecture or context capacity')
    opened = {}
    verified = {}

    def tensor(name, shape, dtype):
        filename = weight_map[name]
        if Path(filename).name != filename:
            raise ValueError('source filename is not a basename')
        if filename not in opened:
            path = args.source / filename
            last_report = 0
            while not path.exists():
                if not args.wait_for_downloads:
                    raise FileNotFoundError(path)
                if time.monotonic() - last_report > 30:
                    print(json.dumps({'waiting_for': filename}), flush=True)
                    last_report = time.monotonic()
                time.sleep(2)
            if path.stat().st_size != files[filename]['size']:
                raise ValueError('source file size mismatch: ' + filename)
            with path.open('rb') as source:
                digest = hashlib.file_digest(source, 'sha256').hexdigest()
            if digest != files[filename]['lfs']['sha256']:
                raise ValueError('source SHA-256 mismatch: ' + filename)
            source = path.open('rb')
            data = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            header_bytes = struct.unpack_from('<Q', data)[0]
            if header_bytes > 16 * 1024 * 1024:
                raise ValueError('oversized safetensors header')
            header = json.loads(data[8:8 + header_bytes])
            opened[filename] = source, data, header, 8 + header_bytes
            verified[filename] = digest
            print(json.dumps({'source_verified': filename, 'sha256': digest}), flush=True)
        source, data, header, offset = opened[filename]
        meta = header[name]
        if meta['shape'] != list(shape) or meta['dtype'] != ('F16' if dtype == '<f2' else 'I32'):
            raise ValueError('unexpected tensor shape/dtype: ' + name)
        start, stop = meta['data_offsets']
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if start < 0 or stop - start != size or offset + stop > len(data):
            raise ValueError('invalid tensor bounds: ' + name)
        return np.frombuffer(data, dtype=dtype, count=size // np.dtype(dtype).itemsize,
                             offset=offset + start).reshape(shape)

    def matrix(layer, expert, projection):
        k, n = (640, 2560) if projection == 'down_proj' else (2560, 640)
        kp = (k + 255) // 256 * 256
        prefix = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.'
        if prefix + 'g_idx' in weight_map:
            raise ValueError('activation-order indices need a separate importer')
        packed = tensor(prefix + 'qweight', (k // 8, n), '<u4')
        zero_words = tensor(prefix + 'qzeros', (k // 128, n // 8), '<u4')
        scales = tensor(prefix + 'scales', (k // 128, n), '<f2')
        if not np.all(zero_words == 0x77777777):
            raise ValueError('nonconstant symmetric zero point: ' + prefix)
        if not np.isfinite(scales).all():
            raise ValueError('nonfinite source scales: ' + prefix)
        source_codes = ((packed[:, None, :] >> np.arange(0, 32, 4, dtype=np.uint32)[None, :, None]) & 15).reshape(k, n).T.astype(np.uint8)
        codes = np.full((n, kp), 8, dtype=np.uint8)
        codes[:, :k] = source_codes
        blocks = codes.reshape(n, kp // 32, 32)
        nibbles = blocks[:, :, :16] | (blocks[:, :, 16:] << 4)
        padded_scales = np.zeros((n, kp // 128), dtype='<f2')
        padded_scales[:, :k // 128] = scales.T
        raw = np.zeros((n, kp // 256, 136), dtype=np.uint8)
        raw[:, :, :4] = padded_scales.view(np.uint8).reshape(n, kp // 256, 4)
        raw[:, :, 4:132] = nibbles.reshape(n, kp // 256, 128)
        # Independently decode the storage layout, checking every source code
        # and every scale bit; this implies equality of the decoded F32 weights.
        stored_nibbles = raw[:, :, 4:132].reshape(n, kp // 32, 16)
        restored = np.empty((n, kp // 32, 32), dtype=np.uint8)
        restored[:, :, :16] = stored_nibbles & 15
        restored[:, :, 16:] = stored_nibbles >> 4
        stored_scales = raw[:, :, :4].copy().reshape(n, kp // 128, 2).view('<u2').reshape(n, kp // 128)
        if not np.array_equal(restored.reshape(n, kp)[:, :k], source_codes):
            raise ValueError('imported codes differ: ' + prefix)
        if not np.array_equal(stored_scales[:, :k // 128], scales.T.view('<u2')):
            raise ValueError('imported scale bits differ: ' + prefix)
        return raw.tobytes()

    parts = [640 * 10 * 136, 640 * 10 * 136, 2560 * 3 * 136]
    record = sum(parts)
    assert record % 4096 == 0
    file_bytes = 4096 + 48 * 512 * record
    temporary = args.output.with_name(args.output.name + '.partial')
    header = bytearray(4096)
    struct.pack_into('<IIQQQQ', header, 0, 0x58435241, 1, int(REVISION[:16], 16), 49, 512, file_bytes)
    for layer in range(48):
        struct.pack_into('<3I4x5Q', header, 40 + layer * 56, AR4, AR4, AR4,
                         *parts, record, 4096 + layer * 512 * record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    layer_hashes = []
    start = time.monotonic()
    with temporary.open('wb') as output:
        output.write(header)
        for layer in range(48):
            digest = hashlib.sha256()
            for expert in range(512):
                for projection in ['gate_proj', 'up_proj', 'down_proj']:
                    raw = matrix(layer, expert, projection)
                    output.write(raw)
                    digest.update(raw)
            output.flush()
            layer_hashes.append(digest.hexdigest())
            print(json.dumps({'layer_complete': layer, 'bytes_written': output.tell(),
                              'sha256': digest.hexdigest(), 'elapsed_s': time.monotonic() - start}), flush=True)
        output.flush(); os.fsync(output.fileno())
    if temporary.stat().st_size != file_bytes:
        raise ValueError('incorrect expert-file length')
    result = {'source_model': info['id'], 'revision': REVISION, 'format': 'AR4_128',
              'gpu_superblock_bytes': 136, 'record_bytes': record, 'file_bytes': file_bytes,
              'validated_weight_count': 48 * 512 * 3 * 2560 * 640,
              'source_sha256': verified, 'layer_sha256': layer_hashes,
              'seconds': time.monotonic() - start}
    temporary.rename(args.output)
    args.output.with_suffix('.import.json').write_text(json.dumps(result, indent=2), newline='')
    print(json.dumps({'import_complete': str(args.output), **result}), flush=True)

if __name__ == '__main__':
    main()
