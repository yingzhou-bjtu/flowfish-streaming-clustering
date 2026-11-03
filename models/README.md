# External Model Files

Model weights are not distributed with this repository.

Place a compatible speaker-embedding ONNX model at:

```text
models/speaker_embedding.onnx
```

Alternatively, pass `SV_ONNX_PATH=/absolute/path/to/model.onnx`.

The expected input is mono 16 kHz audio converted to normalized 80-dimensional
FBank features with shape `[1, frames, 80]`.
