"""
IRIS Reusable Sample Scope Selector Widget
==========================================
Provides a unified sample-scope selector bar (``Active Sample(s)``,
``Selected Samples`` with checkable multi-select dropdown, and ``All Samples``)
and canonical DataFrame filtering across IRIS Viewer, Phenotyper, and Analyzer.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

import pandas as pd
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
    from qtpy.QtWidgets import QAction
except ImportError:  # Qt6 compatibility
    from qtpy.QtGui import QAction  # type: ignore[no-redef]

from iris_common.schema import find_sample_column

SCOPE_ACTIVE: str = "Active Sample(s)"
SCOPE_SELECTED: str = "Selected Samples"
SCOPE_ALL: str = "All Samples"
SCOPE_MODES: tuple[str, ...] = (SCOPE_ACTIVE, SCOPE_SELECTED, SCOPE_ALL)

_SCOPE_MODE_ALIASES: Dict[str, str] = {
    SCOPE_ACTIVE: SCOPE_ACTIVE,
    "active": SCOPE_ACTIVE,
    "active sample": SCOPE_ACTIVE,
    "active samples": SCOPE_ACTIVE,
    "active sample(s)": SCOPE_ACTIVE,
    SCOPE_SELECTED: SCOPE_SELECTED,
    "selected": SCOPE_SELECTED,
    "selected sample": SCOPE_SELECTED,
    "selected samples": SCOPE_SELECTED,
    SCOPE_ALL: SCOPE_ALL,
    "all": SCOPE_ALL,
    "all sample": SCOPE_ALL,
    "all samples": SCOPE_ALL,
}

_SAMPLE_EXT_RE = re.compile(r"(?:\.ome)?\.tiff?$", re.IGNORECASE)
_SAMPLE_ROLE_RE = re.compile(r"(?:_image|_mask)$", re.IGNORECASE)


def _normalize_scope_mode(mode: str) -> str:
    """Validate and resolve a scope mode string or short alias."""
    if not isinstance(mode, str):
        raise ValueError(f"Scope mode must be a string, got {type(mode).__name__}.")
    cleaned = mode.strip()
    if cleaned in _SCOPE_MODE_ALIASES:
        return _SCOPE_MODE_ALIASES[cleaned]
    lowered = _SCOPE_MODE_ALIASES.get(cleaned.lower())
    if lowered is not None:
        return lowered
    raise ValueError(f"Unknown scope mode {mode!r}; expected one of {SCOPE_MODES}.")


def _normalize_sample_name(value: object) -> str:
    """Strip whitespace, TIFF extensions, and ``_image``/``_mask`` suffixes."""
    if value is None:
        return ""
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    text = _SAMPLE_EXT_RE.sub("", text)
    text = _SAMPLE_ROLE_RE.sub("", text)
    return text


def _dedupe_names(names: Sequence[str]) -> List[str]:
    """Return non-empty stripped strings in insertion order without duplicates."""
    seen: Dict[str, None] = {}
    for item in names:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            seen.setdefault(text, None)
    return list(seen.keys())


class SampleScopeBar(QWidget):
    """
    Compact reusable widget for choosing sample analysis/visualization scope.

    Signals
    -------
    scopeChanged : Signal(str, list)
        Emitted with ``(mode, resolved_samples)`` whenever scope mode or checked
        sample selection changes through user interaction or ``set_scope_mode``.
    """

    scopeChanged = Signal(str, list)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        default_mode: str = SCOPE_ALL,
        *,
        label: str = "Scope:",
        compact: bool = False,
        stacked: bool = False,
    ) -> None:
        super().__init__(parent)
        canonical_mode = _normalize_scope_mode(default_mode)
        self._label_text: str = str(label)
        self._compact: bool = bool(compact)
        self._stacked: bool = bool(stacked)

        self._all_samples: List[str] = []
        self._active_samples: List[str] = []
        self._checked_samples: List[str] = []
        self._sample_actions: Dict[str, QAction] = {}
        self._updating: bool = False

        self._build_ui(canonical_mode)
        self._rebuild_menu()
        self._sync_picker_visibility_and_summary()

    def _build_ui(self, default_mode: str) -> None:
        spacing = 4 if self._compact else 6

        self.lbl_prefix: Optional[QLabel] = None
        if self._label_text:
            self.lbl_prefix = QLabel(self._label_text, self)
            self.lbl_prefix.setStyleSheet(
                "color: #E0E0E0; font-size: 12px; background-color: transparent;"
            )

        self.combo_scope = QComboBox(self)
        self.combo_scope.addItems(list(SCOPE_MODES))
        self.combo_scope.setCurrentText(default_mode)
        self.combo_scope.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_scope.setSizePolicy(
            QSizePolicy.Expanding if self._stacked else QSizePolicy.Preferred,
            QSizePolicy.Fixed,
        )
        self.combo_scope.currentTextChanged.connect(self._on_combo_scope_changed)

        self.btn_sample_picker = QPushButton("Choose Samples", self)
        self.btn_sample_picker.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_sample_picker.setStyleSheet(
            "QPushButton { "
            "background-color: #2A2E39; color: #F0F2F5; "
            "border: 1px solid #4B5364; border-radius: 4px; "
            "padding: 3px 8px; min-height: 22px; "
            "font-weight: normal; text-align: left; "
            "} "
            "QPushButton:hover { border: 1px solid #4B82B5; background-color: #323744; } "
            "QPushButton:disabled { color: #777777; background-color: #232832; }"
        )
        self.sample_menu = QMenu(self.btn_sample_picker)
        self.btn_sample_picker.setMenu(self.sample_menu)

        self.lbl_count = QLabel(self)
        self.lbl_count.setStyleSheet(
            "color: #9AA4B2; font-size: 11px; background-color: transparent;"
        )
        self.lbl_count.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        if self._stacked:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(spacing)

            row1 = QHBoxLayout()
            row1.setContentsMargins(0, 0, 0, 0)
            row1.setSpacing(spacing)
            if self.lbl_prefix is not None:
                row1.addWidget(self.lbl_prefix)
            row1.addWidget(self.combo_scope, 1)
            row1.addWidget(self.lbl_count)
            outer.addLayout(row1)
            outer.addWidget(self.btn_sample_picker)
        else:
            row = QHBoxLayout(self)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(spacing)
            if self.lbl_prefix is not None:
                row.addWidget(self.lbl_prefix)
            row.addWidget(self.combo_scope)
            row.addWidget(self.btn_sample_picker, 1)
            row.addWidget(self.lbl_count)

    def _rebuild_menu(self) -> None:
        self._updating = True
        try:
            self.sample_menu.clear()
            self._sample_actions.clear()

            act_all = QAction("Select All", self.sample_menu)
            act_all.setEnabled(bool(self._all_samples))
            act_all.triggered.connect(self._select_all_samples)
            self.sample_menu.addAction(act_all)

            act_clear = QAction("Clear", self.sample_menu)
            act_clear.setEnabled(bool(self._all_samples))
            act_clear.triggered.connect(self._clear_samples)
            self.sample_menu.addAction(act_clear)

            self.sample_menu.addSeparator()

            checked_set = set(self._checked_samples)
            for sample_name in self._all_samples:
                action = QAction(sample_name, self.sample_menu)
                action.setCheckable(True)
                action.setChecked(sample_name in checked_set)
                action.toggled.connect(
                    lambda checked, name=sample_name: self._on_sample_action_toggled(
                        name, checked
                    )
                )
                self.sample_menu.addAction(action)
                self._sample_actions[sample_name] = action
        finally:
            self._updating = False

    def _map_to_known_samples(self, candidates: Sequence[str]) -> List[str]:
        """Match *candidates* against ``self._all_samples`` by exact then stripped stem."""
        cleaned = _dedupe_names(candidates)
        if not self._all_samples:
            return cleaned

        exact_set = set(self._all_samples)
        norm_to_canon: Dict[str, str] = {}
        for sample in self._all_samples:
            norm = _normalize_sample_name(sample)
            if norm and norm not in norm_to_canon:
                norm_to_canon[norm] = sample

        matched: List[str] = []
        for item in cleaned:
            if item in exact_set:
                matched.append(item)
            else:
                norm = _normalize_sample_name(item)
                if norm in norm_to_canon:
                    matched.append(norm_to_canon[norm])
        return list(dict.fromkeys(matched))

    def _sync_picker_visibility_and_summary(self) -> None:
        mode = self.get_scope_mode()
        is_selected_mode = mode == SCOPE_SELECTED
        self.btn_sample_picker.setVisible(is_selected_mode)
        self.btn_sample_picker.setEnabled(bool(self._all_samples))

        resolved = self.resolve_samples()
        total = len(self._all_samples)

        if mode == SCOPE_ALL:
            self.lbl_count.setText(f"All ({total})")
            self.lbl_count.setToolTip(
                ", ".join(self._all_samples) if self._all_samples else "No samples loaded"
            )
        elif mode == SCOPE_ACTIVE:
            self.lbl_count.setText(f"Active ({len(resolved)})")
            self.lbl_count.setToolTip(
                ", ".join(resolved) if resolved else "No active sample"
            )
        else:
            n_checked = len(resolved)
            self.lbl_count.setText(f"{n_checked} selected")
            self.lbl_count.setToolTip(
                ", ".join(resolved) if resolved else "No samples selected"
            )

        if total == 0:
            self.btn_sample_picker.setText("No samples")
            self.btn_sample_picker.setToolTip("No samples available")
        elif len(self._checked_samples) == 0:
            self.btn_sample_picker.setText("Choose samples (0)")
            self.btn_sample_picker.setToolTip("Click to select one or more samples")
        elif len(self._checked_samples) == total:
            self.btn_sample_picker.setText(f"All selected ({total})")
            self.btn_sample_picker.setToolTip(", ".join(self._checked_samples))
        elif len(self._checked_samples) == 1:
            only = self._checked_samples[0]
            self.btn_sample_picker.setText(only)
            self.btn_sample_picker.setToolTip(only)
        else:
            self.btn_sample_picker.setText(
                f"{len(self._checked_samples)} of {total} selected"
            )
            self.btn_sample_picker.setToolTip(", ".join(self._checked_samples))

    def _emit_scope_changed(self) -> None:
        self.scopeChanged.emit(self.get_scope_mode(), self.resolve_samples())

    def _on_combo_scope_changed(self, _mode: str) -> None:
        self._sync_picker_visibility_and_summary()
        if not self._updating:
            self._emit_scope_changed()

    def _on_sample_action_toggled(self, _sample_name: str, _checked: bool) -> None:
        if self._updating:
            return
        self._checked_samples = [
            sample
            for sample in self._all_samples
            if sample in self._sample_actions and self._sample_actions[sample].isChecked()
        ]
        self._sync_picker_visibility_and_summary()
        self._emit_scope_changed()

    def _select_all_samples(self) -> None:
        self._updating = True
        try:
            self._checked_samples = list(self._all_samples)
            for action in self._sample_actions.values():
                action.setChecked(True)
        finally:
            self._updating = False
        self._sync_picker_visibility_and_summary()
        self._emit_scope_changed()

    def _clear_samples(self) -> None:
        self._updating = True
        try:
            self._checked_samples = []
            for action in self._sample_actions.values():
                action.setChecked(False)
        finally:
            self._updating = False
        self._sync_picker_visibility_and_summary()
        self._emit_scope_changed()

    def set_samples(
        self,
        all_samples: Sequence[str],
        active_samples: Optional[Sequence[str]] = None,
        *,
        emit: bool = False,
    ) -> None:
        """Update available sample list while preserving existing checked selections."""
        self._all_samples = _dedupe_names(all_samples)
        if active_samples is not None:
            self._active_samples = _dedupe_names(active_samples)

        if self._all_samples:
            self._checked_samples = self._map_to_known_samples(self._checked_samples)
        else:
            self._checked_samples = []

        self._rebuild_menu()
        self._sync_picker_visibility_and_summary()
        if emit:
            self._emit_scope_changed()

    def set_active_samples(
        self,
        active_samples: Sequence[str],
        *,
        emit: bool = False,
    ) -> None:
        """Update the active sample list used in ``Active Sample(s)`` mode."""
        self._active_samples = _dedupe_names(active_samples)
        self._sync_picker_visibility_and_summary()
        if emit:
            self._emit_scope_changed()

    def get_scope_mode(self) -> str:
        """Return the currently selected scope mode string."""
        return self.combo_scope.currentText()

    def set_scope_mode(self, mode: str, *, emit: bool = True) -> None:
        """Programmatically set the scope mode (accepting canonical names or aliases)."""
        canonical = _normalize_scope_mode(mode)
        if canonical == self.get_scope_mode():
            self._sync_picker_visibility_and_summary()
            return
        if emit:
            self.combo_scope.setCurrentText(canonical)
            return
        self._updating = True
        try:
            self.combo_scope.setCurrentText(canonical)
        finally:
            self._updating = False
        self._sync_picker_visibility_and_summary()

    def get_checked_samples(self) -> List[str]:
        """Return checked sample names in ``Selected Samples`` mode."""
        if self._all_samples:
            allowed = set(self._all_samples)
            return [s for s in self._checked_samples if s in allowed]
        return list(self._checked_samples)

    def set_checked_samples(
        self,
        samples: Sequence[str],
        *,
        emit: bool = False,
    ) -> None:
        """Programmatically update checked samples and synchronize menu checkboxes."""
        self._checked_samples = self._map_to_known_samples(samples)
        checked_set = set(self._checked_samples)
        self._updating = True
        try:
            for name, action in self._sample_actions.items():
                action.setChecked(name in checked_set)
        finally:
            self._updating = False
        self._sync_picker_visibility_and_summary()
        if emit:
            self._emit_scope_changed()

    def resolve_samples(self) -> List[str]:
        """Resolve the effective list of sample names for the current scope mode."""
        mode = self.get_scope_mode()
        if mode == SCOPE_ALL:
            return list(self._all_samples)
        if mode == SCOPE_SELECTED:
            return self.get_checked_samples()

        # SCOPE_ACTIVE
        if self._active_samples:
            resolved = self._map_to_known_samples(self._active_samples)
            if resolved:
                return resolved
        if self._all_samples:
            return [self._all_samples[0]]
        return []

    def filter_dataframe(
        self,
        df: Optional[pd.DataFrame],
        *,
        sample_col: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Filter *df* according to the current scope mode and resolved samples."""
        if df is None:
            return None

        mode = self.get_scope_mode()
        if mode == SCOPE_ALL:
            return df.copy()

        targets = self.resolve_samples()
        if not targets:
            return df.iloc[0:0].copy()

        col = (
            sample_col
            if (sample_col is not None and sample_col in df.columns)
            else find_sample_column(df)
        )
        if col is None:
            return df.copy()

        exact_targets = {str(t).strip() for t in targets}
        norm_targets = {
            _normalize_sample_name(t) for t in targets if _normalize_sample_name(t)
        }

        col_series = df[col]
        non_null = col_series.notna()
        str_vals = col_series.astype("string").fillna("").str.strip()
        norm_vals = str_vals.map(_normalize_sample_name)
        mask = non_null & (str_vals.isin(exact_targets) | norm_vals.isin(norm_targets))
        return df.loc[mask].copy()


__all__ = [
    "SCOPE_ACTIVE",
    "SCOPE_SELECTED",
    "SCOPE_ALL",
    "SCOPE_MODES",
    "SampleScopeBar",
]
