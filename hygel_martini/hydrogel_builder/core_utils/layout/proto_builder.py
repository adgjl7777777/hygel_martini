"""Proto-level backbone/linker utilities."""

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Tuple, Optional

import random

import numpy as np
from hygel_martini.hydrogel_builder.core_utils.templates.linker_loader import LinkerTemplateLibrary
CHAIN_AXIS = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
LINKER_AXIS = np.array([1.0, 0.0, 0.0])


def _weighted_average(lengths: Sequence[float], weights: Sequence[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return float(np.mean(lengths)) if lengths else 0.0
    return float(sum(l * w for l, w in zip(lengths, weights)) / total_weight)


def _normalize_linker_axes(linker_axes: Optional[Sequence[str]]) -> List[str]:
    if linker_axes is None:
        axes = ["x", "x"]
    elif isinstance(linker_axes, str):
        axes = [linker_axes]
    else:
        axes = list(linker_axes)
    cleaned = []
    for axis in axes:
        axis_str = str(axis).lower()
        if axis_str in ("x", "y", "z"):
            cleaned.append(axis_str)
    if not cleaned:
        cleaned = ["x", "x"]
    if len(cleaned) == 1:
        cleaned = [cleaned[0], cleaned[0]]
    return cleaned[:2]


def _compute_linker_length(definition: Dict[str, Any]) -> float:
    internal = definition.get("bonds", [])
    external = definition.get("external_bonds", [])
    total = 0.0
    for bond in internal:
        total += bond.get("length", 0.0)
    for ext in external:
        total += ext.get("length", 0.0)
    return float(total)


@dataclass
class ProtoChain:
    positions: np.ndarray
    types: List[Tuple[str, str]]
    length: float  # scaled length (n segments)
    raw_length: float  # physical length of n-2 beads


@dataclass
class ProtoPlan:
    segment_length: int
    proto_backbone: ProtoChain
    proto_linker: Optional[ProtoChain]
    box_margin: float
    cell_vector: np.ndarray
    medium_size: np.ndarray
    small_size: np.ndarray
    mean_sep: float
    bond_lookup: Dict[Tuple[str, str], Dict[str, Any]]
    sequence_factory: "BackboneSequenceFactory"
    linker_library: LinkerTemplateLibrary | None = None
    linker_span_lookup: Dict[str, float] | None = None

    def box_vector(self, repeats: Tuple[int, int, int]) -> np.ndarray:
        return self.cell_vector * np.array(repeats, dtype=np.float64)


def _build_bond_lookup(bond_rules: Optional[List[Dict[str, Any]]],
                       fallback: float) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not bond_rules:
        return {}
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rule in bond_rules:
        between = rule.get("between", [])
        if len(between) != 2:
            continue
        key = tuple(sorted(between))
        lookup[key] = rule
    return lookup


def _next_backbone_entry(strategy: Dict[str, Any],
                         backbones: List[Dict[str, Any]]):
    strat = (strategy or {}).get('strategy', 'random').lower()
    if strat == 'alternating':
        while True:
            for entry in backbones:
                yield entry
    elif strat == 'block':
        sequence: List[Dict[str, Any]] = []
        blocks = (strategy or {}).get('blocks', [])
        index = {entry.get('id'): entry for entry in backbones}
        for block in blocks:
            entry = index.get(block.get('id'))
            if entry:
                sequence.extend([entry] * block.get('size', 1))
        if not sequence:
            sequence = backbones
        while True:
            for entry in sequence:
                yield entry
    else:
        weights = [entry.get('ratio', 1) for entry in backbones]
        while True:
            yield random.choices(backbones, weights=weights, k=1)[0]


def _build_backbone_sequence(length: int,
                             strategy: Dict[str, Any],
                             backbones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if length <= 0 or not backbones:
        return []
    gen = _next_backbone_entry(strategy, backbones)
    return [next(gen) for _ in range(length)]


def _build_backbone_positions(sequence: List[Dict[str, Any]],
                              bond_lookup: Dict[Tuple[str, str], Dict[str, Any]],
                              fallback: float) -> Tuple[np.ndarray, float]:
    if not sequence:
        return np.zeros((0, 3), dtype=np.float64), 0.0
    positions = []
    current = np.zeros(3, dtype=np.float64)
    positions.append(current.copy())
    total = 0.0
    for idx in range(len(sequence) - 1):
        first = sequence[idx].get('id')
        second = sequence[idx + 1].get('id')
        key = tuple(sorted((first, second)))
        rule = bond_lookup.get(key, {})
        bond_len = rule.get('bond_c0', fallback)
        if bond_len is None or bond_len <= 0:
            bond_len = fallback
        current = current + CHAIN_AXIS * bond_len
        total += bond_len
        positions.append(current.copy())
    return np.array(positions, dtype=np.float64), total


def _chain_geometry_from_sequence(sequence: List[Dict[str, Any]],
                                  segment_length: int,
                                  mean_sep: float,
                                  bond_lookup: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[np.ndarray, float, float]:
    positions, raw_length = _build_backbone_positions(sequence, bond_lookup, mean_sep)
    num_proto_beads = max(len(sequence), 1)
    if positions.size == 0:
        positions = np.zeros((num_proto_beads, 3), dtype=np.float64)
    if raw_length <= 0:
        raw_length = mean_sep * max(num_proto_beads - 1, 1)
    if segment_length > 3 and (segment_length - 3) > 0:
        scale = (segment_length - 1) / float(segment_length - 3)
        scaled_length = raw_length * scale
    else:
        scaled_length = raw_length
    return positions, float(raw_length), float(scaled_length)


def _resolve_block_pattern(strategy: Dict[str, Any],
                           backbones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks = (strategy or {}).get('blocks', [])
    if not blocks:
        return backbones
    index = {entry.get('id'): entry for entry in backbones}
    pattern: List[Dict[str, Any]] = []
    for block in blocks:
        entry = index.get(block.get('id'))
        if not entry:
            continue
        pattern.extend([entry] * max(int(block.get('size', 1)), 1))
    return pattern or backbones


class BackboneSequenceFactory:
    """
    Generates backbone sequences/rescaled coordinates per placement, honoring the requested strategy.
    """

    def __init__(self,
                 segment_length: int,
                 backbone_definitions: List[Dict[str, Any]],
                 strategy: Optional[Dict[str, Any]],
                 mean_sep: float,
                 bond_lookup: Dict[Tuple[str, str], Dict[str, Any]],
                 prototype_sequence: Optional[List[Dict[str, Any]]] = None):
        self.segment_length = segment_length
        self.definitions = backbone_definitions or []
        self.mean_sep = mean_sep
        self.bond_lookup = bond_lookup or {}
        self.num_proto_beads = max(segment_length - 2, 0)
        self.strategy = (strategy or {}).copy()
        self.strategy_name = self.strategy.get('strategy', 'random').lower()
        self.weights = [entry.get('ratio', 1) for entry in self.definitions] or [1]
        self.pattern: List[Dict[str, Any]] | None = None
        if self.strategy_name == 'alternating':
            self.pattern = self.definitions
        elif self.strategy_name == 'block':
            self.pattern = _resolve_block_pattern(self.strategy, self.definitions)
        self.pattern_offset = 0
        self._long_random_cache: List[Dict[str, Any]] | None = None
        if self.strategy_name == 'long_random' and prototype_sequence:
            if len(prototype_sequence) == self.num_proto_beads:
                self._long_random_cache = list(prototype_sequence)

    def _random_sequence(self) -> List[Dict[str, Any]]:
        if self.num_proto_beads <= 0 or not self.definitions:
            return []
        return random.choices(self.definitions, weights=self.weights, k=self.num_proto_beads)

    def next_sequence(self) -> List[Dict[str, Any]]:
        if self.num_proto_beads <= 0:
            return []
        if self.pattern:
            pattern_len = len(self.pattern)
            if pattern_len == 0:
                return []
            start = self.pattern_offset
            sequence = [
                self.pattern[(start + idx) % pattern_len]
                for idx in range(self.num_proto_beads)
            ]
            self.pattern_offset = (self.pattern_offset + 1) % max(pattern_len, 1)
            return sequence
        if self.strategy_name == 'long_random':
            if self._long_random_cache is None or len(self._long_random_cache) != self.num_proto_beads:
                self._long_random_cache = self._random_sequence()
            return list(self._long_random_cache)
        return self._random_sequence()

    def instantiate(self,
                    enforce_unique: bool = False,
                    used_signatures: Optional[set] = None,
                    max_attempts: int = 50) -> Tuple[List[Dict[str, Any]], np.ndarray, float, float]:
        attempt = 0
        sequence: List[Dict[str, Any]] = []
        signature: Tuple[str, ...] = tuple()
        require_unique = enforce_unique and used_signatures is not None and len(self.definitions) > 1
        while True:
            sequence = self.next_sequence()
            signature = tuple(entry.get('id') for entry in sequence)
            if not require_unique or signature not in used_signatures:
                break
            attempt += 1
            if attempt >= max_attempts:
                break
        if require_unique and used_signatures is not None:
            used_signatures.add(signature)
        positions, raw_length, scaled_length = _chain_geometry_from_sequence(sequence, self.segment_length,
                                                                             self.mean_sep, self.bond_lookup)
        return sequence, positions, raw_length, scaled_length
    
    @property
    def definition_count(self) -> int:
        return len(self.definitions)


def build_proto_backbone(segment_length: int,
                         backbone_definitions: List[Dict[str, Any]],
                         mean_sep: float,
                         strategy: Optional[Dict[str, Any]] = None,
                         bond_rules: Optional[List[Dict[str, Any]]] = None,
                         bond_lookup: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None) -> ProtoChain:
    if segment_length < 2:
        raise ValueError("segment_length must be >= 2")

    num_proto_beads = max(segment_length - 2, 1)
    sequence = _build_backbone_sequence(num_proto_beads, strategy or {'strategy': 'random'}, backbone_definitions)
    if bond_lookup is None:
        bond_lookup = _build_bond_lookup(bond_rules, mean_sep)
    positions, raw_length, scaled_length = _chain_geometry_from_sequence(sequence, segment_length, mean_sep, bond_lookup)

    types = []
    for idx, entry in enumerate(sequence):
        atom_name = entry.get('definition', {}).get('atom_name', f"BB{idx}")
        types.append((entry.get('id', f"BB{idx}"), atom_name))

    return ProtoChain(positions=positions,
                      types=types,
                      length=float(scaled_length),
                      raw_length=float(raw_length))


def build_proto_linker(linker_definitions: List[Dict[str, Any]],
                       strategy: Optional[Dict[str, Any]] = None) -> Optional[ProtoChain]:
    if not linker_definitions:
        return None

    strategy = (strategy or {}).get('strategy', 'random').lower()
    if strategy == 'alternating':
        selected = linker_definitions[0]
    elif strategy == 'block':
        selected = linker_definitions[0]
    else:
        weights = [entry.get('ratio', 1) for entry in linker_definitions]
        selected = random.choices(linker_definitions, weights=weights, k=1)[0]

    definition = selected.get('definition', {})
    avg_length = _compute_linker_length(definition)
    beads = definition.get('beads', [])
    num_beads = max(len(beads), 2)
    num_segments = max(num_beads - 1, 1)
    bead_spacing = avg_length / float(num_segments)

    positions = []
    types = []
    for idx in range(num_beads):
        coord = LINKER_AXIS * bead_spacing * idx
        positions.append(coord)
        bead_name = beads[idx].get('name') if idx < len(beads) else f"LN{idx}"
        types.append((definition.get('id', 'LNK'), bead_name or f"LN{idx}"))

    return ProtoChain(positions=np.array(positions, dtype=np.float64),
                      types=types,
                      length=avg_length,
                      raw_length=avg_length)


def prepare_proto_plan(segment_length: int,
                       mean_sep: float,
                       backbone_defs: List[Dict[str, Any]],
                       linker_defs: List[Dict[str, Any]],
                       box_margin: float,
                       backbone_strategy: Optional[Dict[str, Any]] = None,
                       linker_strategy: Optional[Dict[str, Any]] = None,
                       bond_rules: Optional[List[Dict[str, Any]]] = None,
                       linker_library: LinkerTemplateLibrary | None = None,
                       linker_axes: Optional[Sequence[str]] = None) -> ProtoPlan:

    bond_lookup = _build_bond_lookup(bond_rules, mean_sep)
    proto_backbone = build_proto_backbone(segment_length, backbone_defs, mean_sep,
                                          strategy=backbone_strategy, bond_rules=bond_rules,
                                          bond_lookup=bond_lookup)
    proto_linker = build_proto_linker(linker_defs, linker_strategy)

    linker_span_lookup: Dict[str, float] | None = None
    linker_span_values: List[float] = []
    span_weights: List[float] = []
    if linker_library and linker_library.records:
        linker_span_lookup = {}
        for record in linker_library.records:
            span = float(record.template.span_length)
            linker_span_lookup[record.template.id] = span
            linker_span_values.append(span)
            span_weights.append(max(float(record.ratio), 0.0))
    else:
        for entry in linker_defs:
            definition = entry.get('definition', {})
            span = definition.get('span_from_gro')
            if span and span > 0:
                linker_span_values.append(float(span))
                span_weights.append(entry.get('ratio', 1.0))

    if linker_span_values:
        avg_span = _weighted_average(linker_span_values,
                                     span_weights or [1.0] * len(linker_span_values))
        proto_linker = ProtoChain(positions=np.array(proto_linker.positions, dtype=np.float64),
                                  types=list(proto_linker.types),
                                  length=avg_span,
                                  raw_length=avg_span)
    margin = max(box_margin, 0.0)
    small_edge = proto_backbone.length / np.sqrt(3.0)
    small_size = np.array([small_edge, small_edge, small_edge], dtype=np.float64)
    effective_linker_len = proto_linker.length if proto_linker is not None else 0.0
    linker_axes = _normalize_linker_axes(linker_axes)
    base_size = np.array([2.0 * small_edge, 2.0 * small_edge, 2.0 * small_edge], dtype=np.float64)
    for axis in linker_axes:
        if axis == "x":
            base_size[0] += effective_linker_len
        elif axis == "y":
            base_size[1] += effective_linker_len
        elif axis == "z":
            base_size[2] += effective_linker_len
    medium_size = base_size
    cell_vector = medium_size * 2.0
    id_lookup = {entry.get('id'): entry for entry in backbone_defs}
    proto_sequence = []
    for bead_id, _ in proto_backbone.types:
        entry = id_lookup.get(bead_id)
        if entry:
            proto_sequence.append(entry)
    sequence_factory = BackboneSequenceFactory(segment_length, backbone_defs,
                                               backbone_strategy, mean_sep, bond_lookup,
                                               prototype_sequence=proto_sequence)

    return ProtoPlan(segment_length=segment_length,
                     proto_backbone=proto_backbone,
                     proto_linker=proto_linker,
                     box_margin=margin,
                     cell_vector=cell_vector,
                     medium_size=medium_size,
                     small_size=small_size,
                     mean_sep=mean_sep,
                     bond_lookup=bond_lookup,
                     sequence_factory=sequence_factory,
                     linker_library=linker_library,
                     linker_span_lookup=linker_span_lookup)


def describe_proto_summary(segment_length: int,
                           mean_sep: float,
                           backbone_defs: List[Dict[str, Any]],
                           linker_defs: List[Dict[str, Any]],
                           **kwargs):
    proto = prepare_proto_plan(segment_length, mean_sep, backbone_defs, linker_defs, 0.0, **kwargs)
    return {
        'backbone_length': proto.proto_backbone.length,
        'backbone_length_raw': proto.proto_backbone.raw_length,
        'linker_length': proto.proto_linker.length if proto.proto_linker is not None else 0.0,
        'num_backbone_beads': proto.proto_backbone.positions.shape[0],
        'num_linker_beads': proto.proto_linker.positions.shape[0] if proto.proto_linker is not None else 0
    }
    linker_spans = {}
    for entry in linker_defs:
        definition = entry.get('definition', entry)
        internal = definition.get('bonds', [])
        external = definition.get('external_bonds', [])
        span = 0.0
        for bond in internal:
            span += float(bond.get('length', 0.0))
        for ext in external:
            span += float(ext.get('length', 0.0))
        linker_spans[entry.get('id')] = span if span > 0 else proto.proto_linker.length
