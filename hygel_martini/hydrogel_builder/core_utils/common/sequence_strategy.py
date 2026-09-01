"""Shared helpers for applying random/alternating/block strategies to template selections.

Used wherever a build step must repeatedly pick one template out of a weighted
set (polymer backbones, side chains, terminals). Callers wrap their template
entries in ``StrategyRecord`` objects and then draw templates one at a time
with ``TemplateStrategyIterator.next()``.

Invariant: with zero or one record the iterator degenerates to always
returning that single template (strategy ``'single'``), regardless of the
configured strategy.
"""

from dataclasses import dataclass
from typing import List, Optional

import random


@dataclass
class StrategyRecord:
    """One selectable template plus its selection weight.

    Attributes:
        template: The template object to hand back to the caller (opaque to
            this module; may be a dict definition or a loaded template).
        ratio: Relative selection weight (dimensionless). Non-positive values
            are clamped to a weight of 1.0 at draw time.
        template_id: Optional stable identifier used by the ``block`` strategy
            to match ``blocks[].id`` entries against records.
    """

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
        """Prepare the iterator state for the requested strategy.

        Args:
            records: Candidate templates with weights. May be empty.
            strategy_cfg: Optional dict with keys ``strategy`` (``random`` |
                ``alternating`` | ``block``), ``seed`` (for the internal RNG),
                and, for ``block``, ``blocks``: a list of
                ``{'id': ..., 'size': ...}`` entries.
        """
        self.records = records or []
        self.strategy_cfg = strategy_cfg or {}
        self.strategy = (self.strategy_cfg.get('strategy') or 'random').lower()
        self.random_state = random.Random(self.strategy_cfg.get('seed'))
        self._alternating_sequence: List[StrategyRecord] = []
        self._block_sequence: List[StrategyRecord] = []
        self._iterator_index = 0
        self._prepare_sequences()

    def _prepare_sequences(self):
        """Precompute the repeating sequences used by non-random strategies.

        For ``alternating`` the records are cycled in listed order. For
        ``block`` the sequence is built from explicit ``blocks`` entries when
        given; otherwise block sizes are derived from the record ratios
        (normalized by the smallest weight, at least one per record). Unknown
        strategy names fall back to ``random``.
        """
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
        """Return the next template according to the configured strategy.

        Returns:
            The selected record's ``template``, or ``None`` when no records
            were supplied. Weighted-random draws use the seeded internal RNG;
            alternating/block strategies advance a shared position counter.
        """
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
