"""Tests for GenerationReport."""

import contextlib
import io

import numpy as np
import pytest
import xarray as xr

from data_sparsity.generate_data import GenerateData
from data_sparsity.output import GenerationReport


def generate(tmp_path, tag, **kwargs):
    """Run a case and hand back the instance, with its report attached."""
    out = tmp_path / tag
    (out / "nc").mkdir(parents=True)
    (out / "pq").mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
        gen = GenerateData(**kwargs)
        gen.generate(netcdf_filepath=str(out / "nc" / "d.nc"),
                     parquet_filepath=str(out / "pq" / "d.parquet"))
    return gen


class TestStatus:
    """match, adjusted and differs are three different things."""

    def test_matching_values(self):
        report = GenerationReport()
        report.compare("density", "var0", 0.5, 0.5)
        assert report.rows[0]["status"] == "match"
        assert report.differences == []

    def test_validator_edits_are_adjusted_not_differences(self):
        """A report that flags expected rounding trains the reader to ignore it."""
        report = GenerationReport()
        report.compare("num_obs", None, 900, 879, "rounding", adjusted=True)
        assert report.rows[0]["status"] == "adjusted"
        assert report.differences == []

    def test_real_gaps_are_differences(self):
        report = GenerationReport()
        report.compare("density", "var1", 0.14, 1.0, "capped")
        assert report.rows[0]["status"] == "differs"
        assert len(report.differences) == 1
        assert report.rows[0]["evidence"] == "capped"

    def test_evidence_only_appears_when_they_differ(self):
        report = GenerationReport()
        report.compare("density", "var0", 0.5, 0.5, "should not be shown")
        assert report.rows[0]["evidence"] == ""


class TestMeasuredFromTheData:
    """The point of the report: it counts cells, it does not echo state."""

    def test_disagreement_with_the_generator_is_reported(self, tmp_path):
        """Hand it an array that does not match what was asked for."""
        gen = generate(tmp_path, "a", num_obs=500, num_dims=2,
                       ratio_dims=(1, 1), density=0.5, seed=42)
        empty = xr.DataArray(np.full((32, 32), np.nan), dims=("x0", "x1"),
                             name="record")
        report = GenerationReport.from_arrays(gen, empty)
        rows = {r["property"]: r for r in report.rows}
        assert rows["observations"]["achieved"] == 0
        assert rows["observations"]["status"] == "differs"

    def test_clean_run_has_no_differences(self, tmp_path):
        gen = generate(tmp_path, "b", num_obs=900, num_dims=3,
                       ratio_dims=(1, 1, 1), density=[0.4, 0.3, 0.2], seed=8,
                       num_vars=3, overlap=[0.5, 0.3])
        assert gen.report.differences == []
        assert any(r["status"] == "adjusted" for r in gen.report.rows)

    def test_reduced_dimension_saturation_is_caught(self, tmp_path):
        """density is measured against the full grid, so a variable on fewer
        dimensions has its count capped at its own grid and saturates. This
        is silent without the report."""
        gen = generate(tmp_path, "c", num_obs=117600, num_dims=4,
                       ratio_dims=(1, 5, 20, 42),
                       density=[0.7, 0.7, 0.14, 0.14], seed=1, num_vars=4,
                       var_dims=[[0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3], [0, 2, 3]],
                       overlap=[1.0, 1.0, 1.0])
        differing = {(r["property"], r["variable"]) for r in gen.report.differences}
        assert ("density", "var2") in differing
        assert ("density", "var3") in differing
        evidence = [r["evidence"] for r in gen.report.differences]
        assert any("full grid" in e for e in evidence)

    def test_the_two_formats_are_cross_checked(self, tmp_path):
        gen = generate(tmp_path, "d", num_obs=500, num_dims=2,
                       ratio_dims=(1, 1), density=0.5, seed=42)
        rows = [r for r in gen.report.rows if r["property"] == "values in parquet"]
        assert rows and all(r["status"] == "match" for r in rows)

    def test_coverage_is_reported(self, tmp_path):
        gen = generate(tmp_path, "e", num_obs=500, num_dims=2,
                       ratio_dims=(1, 1), density=0.5, seed=42)
        rows = [r for r in gen.report.rows if r["property"] == "coverage"]
        assert rows and rows[0]["achieved"] == "yes"


class TestSerialAndParallelAgree:
    """The chunked path measures in the workers and sums; it must agree."""

    def test_same_report_either_way(self, tmp_path):
        params = dict(num_obs=900, num_dims=3, ratio_dims=(1, 1, 1),
                      density=[0.4, 0.3, 0.2], seed=8, num_vars=3,
                      overlap=[0.5, 0.3])
        serial = generate(tmp_path, "s", **params)
        out = tmp_path / "p"
        (out / "nc").mkdir(parents=True)
        (out / "pq").mkdir(parents=True)
        with contextlib.redirect_stdout(io.StringIO()):
            par = GenerateData(max_obs=400, **params)
            par.generate(netcdf_filepath=str(out / "nc" / "d.nc"),
                         parquet_filepath=str(out / "pq" / "d.parquet"))

        def comparable(report):
            return {(r["property"], r["variable"]): r["achieved"]
                    for r in report.rows
                    if r["property"] in ("observations", "density", "overlap F1",
                                         "coverage")}
        assert comparable(serial.report) == comparable(par.report)


class TestRender:
    def test_render_names_the_differences(self):
        report = GenerationReport()
        report.compare("density", "var1", 0.14, 1.0, "capped at its own grid")
        text = report.render()
        assert "density" in text and "capped at its own grid" in text
        assert "1 of 1 differ" in text

    def test_to_dict_counts_both_kinds(self):
        report = GenerationReport()
        report.compare("num_obs", None, 900, 879, "rounding", adjusted=True)
        report.compare("density", "var1", 0.14, 1.0, "capped")
        summary = report.to_dict()
        assert summary["differences"] == 1
        assert summary["adjusted"] == 1
