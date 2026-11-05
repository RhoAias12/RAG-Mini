"""질문과 가장 비슷한 조각 k개를 찾는다 (코사인 유사도, 임계값 없음)."""
from dataclasses import dataclass

import numpy as np

import config
import embedder
import indexer


@dataclass
class Hit:
    chunk: object      # chunker.Chunk
    score: float       # 코사인 유사도 (-1 ~ 1, 클수록 비슷함)


class Retriever:
    def __init__(self, rebuild=False):
        self.chunks, self.embeddings = indexer.load_index(rebuild=rebuild)

    def search(self, question, k=config.TOP_K):
        q = np.asarray(embedder.embed_query(question), dtype=np.float32)
        q /= np.linalg.norm(q)
        # 조각 벡터는 저장할 때 길이 1로 맞춰 두었으므로 내적 = 코사인 유사도
        scores = self.embeddings @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in top]
