"""
Tests for IRIS canonical schema, shared types, statistical engine, theme, and headless imports.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from iris_common import (
    COL_CELL_ID,
    COL_CELL_TYPE,
    COL_SAMPLE,
    COL_X,
    COL_Y,
    BoundingBox,
    CellTypePanel,
    CellTypeRule,
    MarkerPanel,
    ensure_canonical_dataframe,
    find_sample_column,
    resolve_marker_column,
)
from iris_common.engine.statistics import (
    StatTestResult,
    adjust_p_values,
    format_significance,
    run_group_comparison,
)
from iris_common.widgets.theme import (
    DIALOG_DARK_THEME,
    POPOUT_BUTTON_STYLE,
    PUBLICATION_RC_PRESETS,
    SHARED_DARK_THEME,
    configure_independent_window,
)


class TestCanonicalSchemaAndTypes:
    def test_canonical_constants(self):
        assert COL_SAMPLE == "Image"
        assert COL_CELL_ID == "ObjectNumber"
        assert COL_X == "Pos_X"
        assert COL_Y == "Pos_Y"
        assert COL_CELL_TYPE == "cell_type"

    def test_viewer_core_types_reexport_identity(self):
        from iris_viewer.core import types as viewer_types

        assert viewer_types.MarkerPanel is MarkerPanel
        assert viewer_types.CellTypeRule is CellTypeRule
        assert viewer_types.CellTypePanel is CellTypePanel
        assert viewer_types.BoundingBox is BoundingBox

    def test_marker_panel_and_cell_type_panel_helpers(self):
        mp = MarkerPanel(markers=["DAPI", "CD3", "CD8", "CD20"])
        assert mp.n_channels == 4
        assert mp.index_of("cd3") == 1
        assert mp.subset(["CD8", "Missing", "DAPI"]) == [2, 0]

        panel = CellTypePanel(
            rules=[
                CellTypeRule("CD8 T cell", markers_positive=["CD3", "CD8"], markers_negative=["CD20"]),
                CellTypeRule("B cell", markers_positive=["CD20"], parent="Lymphocyte"),
            ]
        )
        astir_dict = panel.to_astir_dict(mp)
        assert astir_dict == {"cell_types": {"CD8 T cell": ["CD3", "CD8"], "B cell": ["CD20"]}}

        anchor = panel.to_celesta_anchor_matrix(mp)
        assert anchor.loc["CD8 T cell", "CD3"] == 1
        assert anchor.loc["CD8 T cell", "CD20"] == 0
        assert anchor.loc["B cell", "DAPI"] == -1

        box = BoundingBox(2, 3, 10, 15).padded(5, max_h=12, max_w=20)
        assert box.to_tuple() == (0, 0, 12, 20)

    def test_resolve_marker_column(self):
        df = pd.DataFrame(
            {
                "CD3_mean": [1.0, 2.0],
                "CD3_median": [0.9, 1.8],
                "CD8": [3.0, 4.0],
                "FoxP3_max": [0.5, 0.7],
            }
        )
        assert resolve_marker_column(df, "CD3") == "CD3_mean"
        assert resolve_marker_column(df, "cd3", preferred_stat="median") == "CD3_median"
        assert resolve_marker_column(df, "CD8") == "CD8"
        assert resolve_marker_column(df, "CD8_mean") == "CD8"
        assert resolve_marker_column(df, "FoxP3", preferred_stat="mean") == "FoxP3_max"
        assert resolve_marker_column(df, "NonExistent") is None

    def test_find_sample_column_and_ensure_canonical(self):
        raw = pd.DataFrame(
            {
                "sample_id": ["Core01_image.tiff", "Core02.tif", "Core02.tif"],
                "label": [1, 2, 3],
                "voron_x": [10.5, 20.5, 30.5],
                "voron_y": [11.0, 21.0, 31.0],
                "Phenotype": ["Tumor", "T cell", "Macrophage"],
                "CD3_mean": [0.2, 5.1, 0.3],
            }
        )
        assert find_sample_column(raw) == "sample_id"
        canon = ensure_canonical_dataframe(raw)
        assert canon["Image"].tolist() == ["Core01", "Core02", "Core02"]
        assert canon["ObjectNumber"].tolist() == [1, 2, 3]
        assert canon["Pos_X"].tolist() == [10.5, 20.5, 30.5]
        assert canon["Pos_Y"].tolist() == [11.0, 21.0, 31.0]
        assert canon["cell_type"].tolist() == ["Tumor", "T cell", "Macrophage"]
        # Original columns remain preserved
        assert "sample_id" in canon.columns
        assert "CD3_mean" in canon.columns

    def test_ensure_canonical_generates_1_indexed_ids_when_missing(self):
        df = pd.DataFrame({"CD3": [1.0, 2.0, 3.0]})
        canon = ensure_canonical_dataframe(df, default_sample="S1")
        assert canon["Image"].tolist() == ["S1", "S1", "S1"]
        assert canon["ObjectNumber"].tolist() == [1, 2, 3]

        with pytest.raises(ValueError, match="spatial coordinates"):
            ensure_canonical_dataframe(df, require_coordinates=True)


class TestMultipleTestingAndFormatting:
    def test_adjust_p_values_matches_scipy_and_preserves_nan(self):
        ps = np.array([0.001, 0.02, 0.04, 0.5, np.nan])
        bh = adjust_p_values(ps, method="fdr_bh")
        expected_bh = stats.false_discovery_control(ps[:-1], method="bh")
        np.testing.assert_allclose(bh[:-1], expected_bh, rtol=1e-12)
        assert np.isnan(bh[-1])

        bonf = adjust_p_values(ps[:-1], method="bonferroni")
        np.testing.assert_allclose(bonf, np.minimum(1.0, ps[:-1] * 4.0))

        holm = adjust_p_values(np.array([0.01, 0.04, 0.03]), method="holm")
        np.testing.assert_allclose(holm, [0.03, 0.06, 0.06])

        raw_none = adjust_p_values(ps[:-1], method="none")
        np.testing.assert_allclose(raw_none, ps[:-1])

    def test_format_significance_modes(self):
        assert format_significance(1e-5) == "****"
        assert format_significance(5e-4) == "***"
        assert format_significance(5e-3) == "**"
        assert format_significance(0.03) == "*"
        assert format_significance(0.12) == "ns"
        assert format_significance(np.nan) == "ns"
        assert format_significance(None) == "ns"
        assert format_significance(0.0123, mode="p_value") == "p = 0.012"
        assert format_significance(2e-5, mode="q_value") == "q = 2.00e-05"
        assert "**" in format_significance(0.004, mode="stars_and_p")


class TestGroupComparisons:
    @staticmethod
    def _make_cohort_df() -> pd.DataFrame:
        rng = np.random.default_rng(7)
        rows = []
        sample_means = {
            "Responder": [5.0, 5.4, 5.8, 6.2],
            "NonResponder": [1.5, 1.9, 2.2, 2.5],
            "Healthy": [3.3, 3.6, 3.8, 4.0],
        }
        for grp, means in sample_means.items():
            for idx, m in enumerate(means):
                sample_id = f"{grp}_S{idx + 1}"
                vals = rng.normal(loc=m, scale=0.4, size=80)
                for cell_i, val in enumerate(vals, start=1):
                    rows.append(
                        {
                            "Image": sample_id,
                            "ObjectNumber": cell_i,
                            "condition": grp,
                            "patient_id": f"P{idx + 1}",
                            "CD8_mean": float(val),
                        }
                    )
        return pd.DataFrame(rows)

    def test_sample_aggregation_prevents_pseudoreplication(self):
        df = self._make_cohort_df()
        two_grp = df[df["condition"].isin(["Responder", "NonResponder"])]

        sample_res = run_group_comparison(
            two_grp,
            value_col="CD8_mean",
            group_col="condition",
            sample_col="Image",
            aggregate_by_sample=True,
            test="mann_whitney",
        )
        assert len(sample_res) == 1
        r = sample_res[0]
        assert isinstance(r, StatTestResult)
        assert r.aggregation_level == "sample"
        assert r.n1 == 4
        assert r.n2 == 4
        assert r.p_value < 0.05

        cell_res = run_group_comparison(
            two_grp,
            value_col="CD8_mean",
            group_col="condition",
            sample_col="Image",
            aggregate_by_sample=False,
            test="mann_whitney",
        )
        assert cell_res[0].aggregation_level == "cell"
        assert cell_res[0].n1 == 320
        assert cell_res[0].n2 == 320
        assert cell_res[0].p_value < r.p_value

    def test_welch_and_student_t(self):
        df = self._make_cohort_df()
        sub = df[df["condition"].isin(["Responder", "NonResponder"])]

        welch = run_group_comparison(sub, "CD8", "condition", test="welch_t")
        student = run_group_comparison(sub, "CD8_mean", "condition", test="student_t")
        assert welch[0].p_value < 1e-3
        assert student[0].p_value < 1e-3
        assert welch[0].effect_size is not None
        assert student[0].effect_size_name == "cohens_d"

    def test_wilcoxon_paired_by_patient_and_zero_diff_edge_case(self):
        df = self._make_cohort_df()
        sub = df[df["condition"].isin(["Responder", "NonResponder"])]
        paired = run_group_comparison(
            sub,
            value_col="CD8_mean",
            group_col="condition",
            sample_col="Image",
            aggregate_by_sample=True,
            test="wilcoxon_paired",
            paired_by="patient_id",
        )
        assert len(paired) == 1
        assert paired[0].n1 == 4
        assert paired[0].n2 == 4
        assert paired[0].p_value == pytest.approx(0.125)

        identical = pd.DataFrame(
            {
                "Image": ["S1", "S2", "S3", "S1", "S2", "S3"],
                "patient": ["P1", "P2", "P3", "P1", "P2", "P3"],
                "region": ["A", "A", "A", "B", "B", "B"],
                "val": [2.0, 3.0, 4.0, 2.0, 3.0, 4.0],
            }
        )
        res_zero = run_group_comparison(
            identical,
            value_col="val",
            group_col="region",
            test="wilcoxon_paired",
            paired_by="patient",
        )
        assert res_zero[0].p_value == 1.0

    def test_kruskal_and_anova_oneway_posthoc_and_tukey(self):
        df = self._make_cohort_df()

        kruskal_res = run_group_comparison(
            df,
            value_col="CD8_mean",
            group_col="condition",
            test="kruskal",
            correction="holm",
        )
        assert len(kruskal_res) == 3
        assert kruskal_res.omnibus_test == "kruskal"
        assert kruskal_res.omnibus_p_value is not None and kruskal_res.omnibus_p_value < 0.05
        assert all(r.correction_method == "holm" for r in kruskal_res)

        anova_bh = run_group_comparison(
            df,
            value_col="CD8_mean",
            group_col="condition",
            test="anova_oneway",
            correction="fdr_bh",
        )
        assert anova_bh.omnibus_test == "anova_oneway"
        assert anova_bh.omnibus_p_value is not None and anova_bh.omnibus_p_value < 1e-4

        anova_tukey = run_group_comparison(
            df,
            value_col="CD8_mean",
            group_col="condition",
            test="anova_oneway",
            correction="tukey",
        )
        assert len(anova_tukey) == 3
        assert all(r.test_name == "anova_oneway_tukey" for r in anova_tukey)
        assert all(r.q_value < 0.05 for r in anova_tukey)

        df_out = anova_tukey.to_dataframe()
        assert len(df_out) == 3
        assert {"group1", "group2", "p_value", "q_value", "significance_label"}.issubset(
            df_out.columns
        )

    def test_pairs_subset_and_aggregation_level_override(self):
        df = self._make_cohort_df()
        res = run_group_comparison(
            df,
            value_col="CD8_mean",
            group_col="condition",
            pairs=[("Responder", "NonResponder")],
            aggregation_level="cell",
        )
        assert len(res) == 1
        assert res[0].group1 == "Responder"
        assert res[0].group2 == "NonResponder"
        assert res[0].aggregation_level == "cell"
        assert res[0].n1 == 320


class TestThemeAndHeadlessIsolation:
    def test_theme_tokens_present(self):
        assert "#1f232b" in SHARED_DARK_THEME
        assert "#1f232b" in DIALOG_DARK_THEME
        assert "#2D5F8B" in POPOUT_BUTTON_STYLE
        assert "Classic White" in PUBLICATION_RC_PRESETS
        assert PUBLICATION_RC_PRESETS["Classic White"]["pdf.fonttype"] == 42

    def test_configure_independent_window(self, qapp):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDialog

        dlg = QDialog()
        configure_independent_window(dlg)
        assert bool(dlg.windowFlags() & Qt.Window)
        dlg.close()

    def test_headless_imports_without_napari(self):
        cmd = (
            "import sys; sys.modules['napari'] = None; "
            "import iris_common, iris_common.engine, iris_common.widgets, "
            "iris_common.widgets.theme, iris_phenotyper, iris_analyzer; "
            "assert 'napari' not in sys.modules or sys.modules['napari'] is None; "
            "print('HEADLESS_OK')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", cmd],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "HEADLESS_OK" in proc.stdout
