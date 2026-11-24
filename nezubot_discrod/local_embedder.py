"""Local ONNX embedder for e5 models that fastembed's registry lacks.

Implements the small surface DenseIndex relies on: ``embed(texts)``
returning one float32 vector per text (mean-pooled over the attention
mask, unnormalized — DenseIndex normalizes and validates), and
``model.tokenizer`` so batch ordering can use true token lengths.
"""
import os
from types import SimpleNamespace

import numpy as np


class LocalE5Embedder:
    def __init__(self, model_dir, *, threads=4, max_length=128):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = os.fspath(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"), options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {entry.name for entry in self.session.get_inputs()}
        tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tokenizer.enable_truncation(max_length=max_length)
        tokenizer.enable_padding()
        self.tokenizer = tokenizer
        self.model = SimpleNamespace(tokenizer=tokenizer)
        self.max_length = max_length

    def embed(self, texts):
        texts = list(texts)
        if not texts:
            return []
        encodings = self.tokenizer.encode_batch(texts)
        width = max(len(entry.ids) for entry in encodings)
        input_ids = np.zeros((len(texts), width), dtype=np.int64)
        attention = np.zeros((len(texts), width), dtype=np.int64)
        for index, entry in enumerate(encodings):
            input_ids[index, :len(entry.ids)] = entry.ids
            attention[index, :len(entry.attention_mask)] = entry.attention_mask
        feeds = {"input_ids": input_ids, "attention_mask": attention}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self.session.run(None, feeds)[0]
        mask = attention[..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return [row.astype(np.float32) for row in pooled]
