"""Shared helpers for applying random/alternating/block strategies to template selections."""

from dataclasses import dataclass
from typing import List, Optional

import random


@dataclass
class StrategyRecord:
    template: object
    ratio: float
    template_id: Optional[str] = None


class TemplateStrategyIterator:
    """
    Iterates over template records following the requested strategy.
    Supported strategies: random, alternating, block. Defaults to random.
    """

    def __init__(self,
                 records: List[StrategyRecord],
                 strategy_cfg: Optional[dict] = None):
        self.records = records or []
        self.strategy_cfg = strategy_cfg or {}
        self.strategy = (self.strategy_cfg.get('strategy') or 'random').lower()
        self.random_state = random.Random(self.strategy_cfg.get('seed'))
        self._alternating_sequence: List[StrategyRecord] = []
        self._block_sequence: List[StrategyRecord] = []
        self._iterator_index = 0
        self._prepare_sequences()

    def _prepare_sequences(self):
        if len(self.records) <= 1:
            self.strategy = 'single'
            return

        if self.strategy == 'alternating':
            self._alternating_sequence = list(self.records)
        elif self.strategy == 'block':
            blocks = self.strategy_cfg.get('blocks') or []
            lookup = {rec.template_id or getattr(rec.template, 'id', None): rec for rec in self.records}
            sequence: List[StrategyRecord] = []
            if blocks:
                for block in blocks:
                    target = lookup.get(block.get('id'))
                    if target:
                        count = max(int(block.get('size', 1)), 1)
                        sequence.extend([target] * count)
            if not sequence:
                weights = [max(float(rec.ratio), 0.0) or 1.0 for rec in self.records]
                min_weight = min(weights)
                counts = [max(1, int(round(w / min_weight))) for w in weights]
                for rec, count in zip(self.records, counts):
                    sequence.extend([rec] * count)
            self._block_sequence = sequence
        else:
            self.strategy = 'random'

    def next(self):
        if not self.records:
            return None
        if len(self.records) == 1 or self.strategy == 'single':
            return self.records[0].template

        if self.strategy == 'random':
            weights = [max(float(rec.ratio), 0.0) or 1.0 for rec in self.records]
            return self.random_state.choices(
                [rec.template for rec in self.records],
                weights=weights,
                k=1
            )[0]

        if self.strategy == 'alternating' and self._alternating_sequence:
            template = self._alternating_sequence[self._iterator_index % len(self._alternating_sequence)]
            self._iterator_index += 1
            return template.template

        if self.strategy == 'block' and self._block_sequence:
            template = self._block_sequence[self._iterator_index % len(self._block_sequence)]
            self._iterator_index += 1
            return template.template

        # Fallback to random if sequences are missing
        weights = [max(float(rec.ratio), 0.0) or 1.0 for rec in self.records]
        return self.random_state.choices(
            [rec.template for rec in self.records],
            weights=weights,
            k=1
        )[0]
