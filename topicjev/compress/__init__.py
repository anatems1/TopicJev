from __future__ import annotations

from topicjev.compress.base import Compressor
from topicjev.compress.causal import CausalCompressor
from topicjev.compress.lingua import LinguaCompressor
from topicjev.compress.seq2seq import CNNCompressor, Seq2SeqCompressor

__all__ = [
    "Compressor",
    "CNNCompressor",
    "Seq2SeqCompressor",
    "CausalCompressor",
    "LinguaCompressor",
]
