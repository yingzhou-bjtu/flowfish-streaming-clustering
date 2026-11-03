"""Speaker embedding and HDBSCAN clustering with ONNX Runtime."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np

from .cluster_hdbscan import cluster_speakers as _cluster_speakers_impl

_DEFAULT_ONNX_NAME = "speaker_embedding.onnx"
BackendName = Literal["auto", "onnx"]


def _project_root() -> Path:
    """Project root containing local runtime state."""
    return Path(__file__).resolve().parents[1]


def default_sv_offline_root() -> Path:
    return _project_root() / "models"


def default_sv_onnx_path(offline_root: Path | None = None) -> Path:
    env = os.environ.get("SV_ONNX_PATH", "").strip()
    if env:
        return Path(env).resolve()
    root = Path(offline_root) if offline_root is not None else default_sv_offline_root()
    candidates = (root / _DEFAULT_ONNX_NAME,)
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return candidates[0].resolve()


def resolve_backend(name: str, *, offline_root: Path | None = None) -> BackendName:
    n = (name or "onnx").strip().lower()
    if n == "auto":
        n = os.environ.get("SV_BACKEND", "onnx").strip().lower() or "onnx"
    if n != "onnx":
        raise ValueError(f"unsupported backend: {name!r}")
    onnx_p = default_sv_onnx_path(offline_root)
    if n == "onnx" and not onnx_p.is_file():
        raise FileNotFoundError(f"缺少 ONNX 模型: {onnx_p}")
    return "onnx"


def _l2_row_normalize(embeddings: np.ndarray) -> np.ndarray:
    x = np.asarray(embeddings, dtype=np.float64)
    row_n = np.linalg.norm(x, axis=1, keepdims=True)
    row_n = np.maximum(row_n, 1e-12)
    return x / row_n


@dataclass
class ClusterBatchResult:
    labels: np.ndarray
    meta: Dict[str, Any]
    hdbscan_was_noise: np.ndarray
    #: HDBSCAN 原始簇号，噪声为负一。labels 是重标后的编号。
    dense_labels: np.ndarray


class SpeakerEmbedCluster:
    """Speaker embedding and clustering engine."""

    def __init__(
        self,
        offline_root: Path | None = None,
        sv_model_id: str = "custom-onnx",
        device: str = "auto",
        backend: str = "auto",
        onnx_path: Path | None = None,
        ort_threads: int = 0,
    ) -> None:
        self.offline_root = (
            Path(offline_root) if offline_root is not None else default_sv_offline_root()
        )
        self.sv_model_id = sv_model_id
        self.device_str = str(device)
        self.backend: BackendName = resolve_backend(backend, offline_root=self.offline_root)
        self.onnx_path = (
            Path(onnx_path).resolve()
            if onnx_path is not None
            else default_sv_onnx_path(self.offline_root)
        )
        self.ort_threads = int(ort_threads)
        self._onnx: Any = None

    def _ensure_onnx(self) -> None:
        if self._onnx is not None:
            return
        from .onnx_backend import OnnxSpeakerEmbedder

        self._onnx = OnnxSpeakerEmbedder(
            self.onnx_path, ort_threads=self.ort_threads
        )

    def _ensure(self) -> None:
        self._ensure_onnx()

    @property
    def cev(self) -> Any:
        raise RuntimeError("Public runtime uses the built-in HDBSCAN implementation")

    def embed_array(self, wav_16k_mono: np.ndarray) -> np.ndarray:
        """Convert a 16 kHz mono waveform to an embedding."""
        self._ensure()
        wav = np.asarray(wav_16k_mono, dtype=np.float32).reshape(-1)
        return self._onnx.embed_wav_numpy(wav)

    def embed_file(self, wav_path: str | Path) -> np.ndarray:
        self._ensure()
        return self._onnx.embed_wav_path(wav_path)

    def embed_files(self, paths: Sequence[str | Path]) -> np.ndarray:
        if not paths:
            return np.zeros((0, 0), dtype=np.float32)
        rows = [self.embed_file(p) for p in paths]
        return np.stack(rows, axis=0)

    def cluster_speakers(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
    ) -> ClusterBatchResult:
        self._ensure()
        embs = np.asarray(embeddings, dtype=np.float64)
        n = int(embs.shape[0])
        if n == 0:
            return ClusterBatchResult(
                np.zeros(0, dtype=np.int32),
                {"clustering_backend": "none", "skipped": True, "reason": "empty"},
                np.zeros(0, dtype=bool),
                np.zeros(0, dtype=np.int32),
            )
        if n == 1:
            # 只有一条时没法聚类，标成噪声，避免首条就输出编号。
            return ClusterBatchResult(
                np.zeros(1, dtype=np.int32),
                {"clustering_backend": "trivial", "n_distinct_pred_clusters": 1},
                np.ones(1, dtype=bool),
                np.array([-1], dtype=np.int32),
            )
        normed = _l2_row_normalize(embs)
        lab, meta, noise, lab_dense = _cluster_speakers_impl(
            normed,
            int(min_cluster_size),
            int(min_samples),
            float(cluster_selection_epsilon),
        )
        md = dict(meta) if isinstance(meta, dict) else {}
        md["embedding_preprocess"] = "l2_row_normalize_for_hdbscan"
        md["hdbscan_effective_geometry"] = "euclidean_on_unit_rows (cosine-related)"
        md["sv_inference_backend"] = self.backend
        return ClusterBatchResult(
            np.asarray(lab, dtype=np.int32, order="C"),
            md,
            np.asarray(noise, dtype=bool, order="C"),
            np.asarray(lab_dense, dtype=np.int32, order="C"),
        )

    def assign_files(
        self,
        wav_paths: Sequence[str | Path],
        *,
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], ClusterBatchResult]:
        paths = [Path(p) for p in wav_paths]
        E = self.embed_files(paths)
        cr = self.cluster_speakers(
            E,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        recs: List[Dict[str, Any]] = []
        for i, p in enumerate(paths):
            emb = E[i]
            recs.append(
                {
                    "path": str(p.resolve()),
                    "pred_speaker_cluster": int(cr.labels[i]),
                    "hdbscan_was_noise": bool(cr.hdbscan_was_noise[i]),
                }
            )
        return E, recs, cr
