from __future__ import annotations
from collections import defaultdict
from math import ceil
from typing import Tuple, Callable, List

from abc import ABC, abstractmethod
from typing import List

ENCODING = "UTF-8"

class CompressorselfAdd(ABC):

    @abstractmethod
    def fit(self, texts: List[str]) -> CompressorselfAdd:
        raise NotImplementedError()

    @abstractmethod
    def get_compressed_len(self, text: str):
        raise NotImplementedError()

    @abstractmethod
    def dictionary_size(self) -> int:
        raise NotImplementedError()


class CompressorClassifier:

    def __init__(self, compressor_provider: Callable[[], CompressorselfAdd], similar=1, num_compressors_per_class=1):
        self.compressor_provider = compressor_provider
        if similar < 1:
            raise ValueError("Invalid top_k value. Correct value is 1. Cheat is 2 or more.")
        self.top_k = similar
        self.num_compressors_per_class = num_compressors_per_class

    # train_pair is a list of [(label, observation), ...
    def fit(self, train_pair: List[Tuple[str, str]]):
        # concatenate strings that have the same labels
        label_to_texts = defaultdict(list)
        for label, observation in train_pair:
            label_to_texts[label].append(observation)

        self.label_to_compressors = {}
        for label, texts in label_to_texts.items():
            compressors = []
            step = ceil(len(texts) / self.num_compressors_per_class)
            for i in range(0, len(texts), step):
                compressor = self.compressor_provider().fit(texts[i:i + step])
                compressors.append(compressor)
            self.label_to_compressors[label] = compressors

    def predict(self, text):
        label_to_scores = {label: [c.get_compressed_len(text) for c in compressors] for label, compressors in
                           self.label_to_compressors.items()}

        label_to_score = {label: sum(scores) for label, scores in label_to_scores.items()}
        res = []
        for _ in range(self.top_k):
            predicted = min(label_to_score, key=label_to_score.get)
            label_to_score.pop(predicted)
            res.append(predicted)
        return res

    def dictionaries_size(self) -> int:
        s = 0
        for compressors in self.label_to_compressors.values():
            for c in compressors:
                s += c.dictionary_size()
        return s
