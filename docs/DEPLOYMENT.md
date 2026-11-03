# Optional Streaming Service

This repository includes a generic HTTP service around the streaming
speaker-clustering pipeline. The service is independent of the paper
reproduction path in `research/hdbscan_streaming/`.

## Requirements

- Linux with Python 3.10 or newer.
- A C++ build toolchain for `hdbscan`.
- `libsndfile` for WAV I/O.
- A compatible speaker-embedding ONNX model supplied by the operator.

The repository does not redistribute model weights or deployment credentials.

## Model Contract

- Audio input: mono, 16 kHz `float32`.
- Features: 80-dimensional normalized FBank frames.
- ONNX input: `float32[1, frames, 80]`.
- ONNX output: one floating-point speaker embedding per input.

Place the model at `models/speaker_embedding.onnx`, or set `SV_ONNX_PATH` to an
absolute path.

## Start

```bash
./setup_venv.sh
SV_ONNX_PATH=/absolute/path/speaker_embedding.onnx ./start_api.sh
curl -fsS http://127.0.0.1:8765/health
```

Useful environment settings:

| Variable | Purpose |
|---|---|
| `SV_HOST` | Listen address |
| `SV_PORT` | Listen port |
| `SV_ONNX_PATH` | ONNX model path |
| `SV_ORT_THREADS` | ONNX Runtime thread count |
| `SV_API_LOG` | Launcher log path |
| `SV_API_PIDFILE` | PID file used for guarded restart |

## Scope Boundary

The public repository contains reusable software and sanitized run
instructions. Site-specific hardware configuration, internal host paths,
restricted model acquisition details, full operational runbooks, and acceptance
records are maintained in a separate private deployment repository.
