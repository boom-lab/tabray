"""What was asked for, what was produced, and where the two differ.

Measured from the generated arrays rather than from the generator's own
bookkeeping. A report that re-read ``self.var_num_obs`` would agree with a bug
in placement; one that counts occupied cells will not.

See ``docs/generation_report_plan.md``.
"""

from typing import List, Optional, Union

import numpy as np
import xarray as xr

MATCH = "match"
ADJUSTED = "adjusted"  # the validator changed the input and said so
DIFFERS = "differs"


class GenerationReport:
    """A table of requested against achieved, one row per property."""

    #: Relative tolerance for treating two numbers as the same.
    TOLERANCE = 1e-9

    def __init__(self) -> None:
        self.rows: List[dict] = []

    # ---------------------------------------------------------------- rows
    def add(
        self,
        prop: str,
        variable: Optional[str],
        requested,
        achieved,
        status: str,
        evidence: str = "",
    ) -> None:
        """Record one comparison."""
        self.rows.append(
            {
                "property": prop,
                "variable": variable or "",
                "requested": requested,
                "achieved": achieved,
                "status": status,
                "evidence": evidence,
            }
        )

    def compare(
        self,
        prop,
        variable,
        requested,
        achieved,
        evidence="",
        adjusted=False,
        tolerance=None,
    ) -> bool:
        """Add a row, deciding the status by comparing the two numbers."""
        if requested is None:
            self.add(prop, variable, "-", achieved, MATCH)
            return True
        same = self._close(requested, achieved, tolerance)
        status = MATCH if same else (ADJUSTED if adjusted else DIFFERS)
        self.add(
            prop,
            variable,
            requested,
            achieved,
            status,
            "" if same else evidence,
        )
        return same

    def _close(self, a, b, tolerance=None) -> bool:
        tolerance = self.TOLERANCE if tolerance is None else tolerance
        try:
            return bool(np.isclose(float(a), float(b), rtol=tolerance, atol=tolerance))
        except (TypeError, ValueError):
            return a == b

    @property
    def differences(self) -> List[dict]:
        """Rows that did not come out as asked, validator edits excluded."""
        return [r for r in self.rows if r["status"] == DIFFERS]

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "differences": len(self.differences),
            "adjusted": sum(1 for r in self.rows if r["status"] == ADJUSTED),
        }

    # ------------------------------------------------------------- measure
    @staticmethod
    def occupancy(data: Union[xr.DataArray, xr.Dataset]) -> dict:
        """Occupancy mask per variable."""
        if isinstance(data, xr.DataArray):
            return {data.name or "var0": data.notnull()}
        return {name: data[name].notnull() for name in data.data_vars}

    @staticmethod
    def requested_density(gen, index: int) -> Optional[float]:
        """The density as the caller gave it, before validation rounded it.

        - one variable: the constructor's density, or 1 - sparsity; validation
          rewrites gen.var_densities to the rounded grid's figure
        - several: gen.var_densities[index], which keeps the requested value

        Compared at the default tolerance, so the validator's rounding always
        reads "differs".
        """
        densities = getattr(gen, "var_densities", None)
        if densities is None or index >= len(densities):
            return None
        if len(densities) > 1:
            return float(densities[index])
        requested = getattr(gen, "_requested", {})
        if requested.get("density") is not None:
            return float(np.max(requested["density"]))
        if requested.get("sparsity") is not None:
            return 1.0 - float(np.min(requested["sparsity"]))
        return float(densities[index])

    @staticmethod
    def _add_coverage(report, name: str, unused: List[str]) -> None:
        """Coverage row for the reference variable.

        The LHS stage uses every coordinate of every axis for var0, so an
        unused one means placement broke that guarantee.
        """
        report.add(
            "coverage",
            name,
            "every coordinate used",
            "yes" if not unused else f"unused on {','.join(unused)}",
            MATCH if not unused else DIFFERS,
            (
                ""
                if not unused
                else "the LHS stage should use every coordinate of every "
                "axis for the reference variable"
            ),
        )

    @staticmethod
    def _add_minimum_density(report, gen) -> None:
        """Which minimum-density bound applied, and its value.

        - scattered: max(shape) / prod(shape), the LHS covering every axis
        - padded: (lines + n_padded - 1) / prod(shape), one observation per
          line plus one line running the full length
        """
        bound = getattr(gen, "density_zero", None)
        if bound is None:
            return
        if getattr(gen, "layout", "scattered") == "padded":
            rule = "padded: (lines + n_padded - 1) / sites"
        else:
            rule = "scattered: max(shape) / sites"
        report.add("minimum density", None, rule, round(float(bound), 6), MATCH)

    @staticmethod
    def project(mask: xr.DataArray, keep) -> xr.DataArray:
        drop = [d for d in mask.dims if d not in keep]
        return mask.any(dim=drop) if drop else mask

    @classmethod
    def from_arrays(
        cls,
        gen,
        data: Union[xr.DataArray, xr.Dataset],
        dataframe=None,
    ) -> "GenerationReport":
        """Build the report by measuring what was generated.

        Args:
            gen: The GenerateData instance, for what was requested
            data: The DataArray or Dataset that was written
            dataframe: The tabular form, to cross-check the row count

        Returns:
            The report
        """
        report = cls()
        masks = cls.occupancy(data)
        names = list(masks)
        requested = getattr(gen, "_requested", {})

        # --- grid -------------------------------------------------------
        report.compare(
            "num_obs",
            None,
            requested.get("num_obs"),
            int(gen.num_obs),
            "validation rounds the grid to whole coordinates",
            adjusted=True,
        )
        cls._add_minimum_density(report, gen)

        # --- per variable -----------------------------------------------
        for index, name in enumerate(names):
            mask = masks[name]
            occupied = int(mask.values.sum())
            grid = int(np.prod(mask.shape))
            wanted = (
                int(gen.var_num_obs[index])
                if getattr(gen, "var_num_obs", None) is not None
                and index < len(gen.var_num_obs)
                else None
            )
            evidence = ""
            if wanted is not None and wanted >= grid:
                evidence = (
                    f"the count was capped at this variable's own grid "
                    f"of {grid:,} cells"
                )
            report.compare("observations", name, wanted, occupied, evidence)

            wanted_density = cls.requested_density(gen, index)
            # The netCDF variable holds only the dimensions it varies along,
            # so the mask size is the variable's own grid, which is what a
            # density is a share of.
            achieved_density = occupied / grid if grid else 0.0
            report.compare(
                "density",
                name,
                wanted_density,
                round(achieved_density, 6),
                "occupied cells over this variable's own grid; validation "
                "rounds the grid to whole coordinates",
            )

            # Coverage is guaranteed for the reference only, by the LHS stage;
            # the other variables are placed without that promise.
            if index == 0:
                unused = [
                    str(dim)
                    for dim in mask.dims
                    if not mask.any(dim=[d for d in mask.dims if d != dim]).all()
                ]
                cls._add_coverage(report, name, unused)

        # --- overlap ------------------------------------------------------
        targets = getattr(gen, "overlap_target", None)
        if targets is not None and len(names) > 1:
            targets = np.atleast_1d(targets)
            reference = names[0]
            for index, name in enumerate(names[1:]):
                if index >= len(targets):
                    break
                target = targets[index]
                if target is None or (isinstance(target, str)):
                    continue
                shared = [d for d in masks[name].dims if d in masks[reference].dims]
                a = cls.project(masks[reference], shared)
                b = cls.project(masks[name], shared)
                denominator = int(a.values.sum())
                achieved = (
                    int((a & b).values.sum()) / denominator if denominator else 0.0
                )
                reduced = len(masks[name].dims) < len(masks[reference].dims)
                step = 1 / denominator if denominator else 0
                evidence = (
                    "this variable drops a dimension, so its overlap follows "
                    "its own density (docs/explainer_multivar.md)"
                    if reduced
                    else f"the reference has {denominator:,} cells, so F1 moves "
                    f"in steps of {step:.4g} and the target falls between two"
                )
                report.compare(
                    "overlap F1",
                    name,
                    round(float(target), 6),
                    round(achieved, 6),
                    evidence,
                    tolerance=step or 1e-9,
                )

        # --- the two formats agree ----------------------------------------
        # Per variable rather than on the row count: variables on different
        # dimensions occupy different numbers of rows, and a union of their
        # masks would broadcast the smaller ones across the dropped axes.
        if dataframe is not None:
            for index, name in enumerate(names):
                column = (
                    name
                    if name in dataframe.columns
                    else ("record" if "record" in dataframe.columns else None)
                )
                if column is None:
                    continue
                report.compare(
                    "values in parquet",
                    name,
                    int(masks[name].values.sum()),
                    int(dataframe[column].notna().sum()),
                    "the two formats disagree on how many values were written",
                )
        return report

    # ------------------------------------------------- the chunked path
    @staticmethod
    def measure_chunk(records, var_dims_indices, dim_split, stratum_offset):
        """Counts a worker returns so the parent can assemble the report.

        Overlap never spans strata, so intersections and occupancy counts add
        up across chunks. Coverage is a union rather than a sum, so it travels
        as one boolean vector per axis -- cheap, and it avoids re-reading the
        output.

        Args:
            records: Mapping of variable name to its chunk-shaped array
            var_dims_indices: Dimensions each variable varies along
            dim_split: Dimension the chunks run along
            stratum_offset: Index of this chunk's first stratum

        Returns:
            Plain dict, picklable, no arrays larger than an axis
        """
        names = sorted(records)
        masks = {name: ~np.isnan(records[name]) for name in names}
        summary = {
            "occupied": {},
            "axis_used": {},
            "intersect": {},
            "ref_proj": {},
            "offset": int(stratum_offset),
        }

        for name in names:
            summary["occupied"][name] = int(masks[name].sum())

        # coverage is checked for the reference only (see from_arrays)
        reference_mask = masks[names[0]]
        summary["axis_used"][names[0]] = {
            dim: reference_mask.any(
                axis=tuple(d for d in range(reference_mask.ndim) if d != dim),
            ).tolist()
            for dim in range(reference_mask.ndim)
        }

        reference = names[0]
        for index, name in enumerate(names[1:], start=1):
            shared = sorted(set(var_dims_indices[0]) & set(var_dims_indices[index]))
            drop = tuple(d for d in range(masks[name].ndim) if d not in shared)
            a = masks[reference].any(axis=drop) if drop else masks[reference]
            b = masks[name].any(axis=drop) if drop else masks[name]
            summary["intersect"][name] = int((a & b).sum())
            summary["ref_proj"][name] = int(a.sum())
        return summary

    @classmethod
    def from_chunks(cls, gen, summaries) -> "GenerationReport":
        """Assemble the report from what the workers measured.

        Args:
            gen: The GenerateData instance
            summaries: One measure_chunk result per chunk

        Returns:
            The report
        """
        report = cls()
        requested = getattr(gen, "_requested", {})
        report.compare(
            "num_obs",
            None,
            requested.get("num_obs"),
            int(gen.num_obs),
            "validation and chunk rounding both move this",
            adjusted=True,
        )
        cls._add_minimum_density(report, gen)
        if not summaries:
            return report

        names = sorted(summaries[0]["occupied"])
        shape = [int(s) for s in gen.shape]
        for index, name in enumerate(names):
            occupied = sum(s["occupied"][name] for s in summaries)
            wanted = (
                int(gen.var_num_obs[index]) if index < len(gen.var_num_obs) else None
            )
            report.compare("observations", name, wanted, occupied, "")

            # Own grid: the dimensions this variable varies along. Chunk
            # arrays keep every dimension, so it cannot come from the mask.
            grid = int(np.prod([shape[d] for d in gen.var_dims_indices[index]]))
            report.compare(
                "density",
                name,
                cls.requested_density(gen, index),
                round(occupied / grid, 6),
                "occupied cells over this variable's own grid; validation and "
                "chunking round the counts",
            )

            # coverage, reference only: union the per-axis vectors, the split
            # axis placed by each chunk's offset
            if index == 0:
                unused = []
                for dim in range(len(shape)):
                    seen = np.zeros(shape[dim], dtype=bool)
                    for s in summaries:
                        local = np.asarray(s["axis_used"][name][dim], dtype=bool)
                        if dim == gen.dim_split:
                            seen[s["offset"] : s["offset"] + local.size] |= local
                        else:
                            seen |= local
                    if not seen.all():
                        unused.append(f"x{dim}")
                cls._add_coverage(report, name, unused)

        targets = getattr(gen, "overlap_target", None)
        if targets is not None and len(names) > 1:
            targets = np.atleast_1d(targets)
            for index, name in enumerate(names[1:]):
                if index >= len(targets) or targets[index] is None:
                    continue
                if isinstance(targets[index], str):
                    continue
                shared = sum(s["intersect"].get(name, 0) for s in summaries)
                denominator = sum(s["ref_proj"].get(name, 0) for s in summaries)
                achieved = shared / denominator if denominator else 0.0
                step = 1 / denominator if denominator else 0
                report.compare(
                    "overlap F1",
                    name,
                    round(float(targets[index]), 6),
                    round(achieved, 6),
                    f"the reference has {denominator:,} cells, so F1 "
                    f"moves in steps of {step:.4g}",
                    tolerance=step or 1e-9,
                )
        return report

    # -------------------------------------------------------------- render
    def render(self) -> str:
        """The table, as printed at the end of a run."""
        if not self.rows:
            return "Generation report: nothing measured."
        width = max(len(str(r["property"])) for r in self.rows)
        lines = ["Generation report (requested against achieved):"]
        for row in self.rows:
            mark = {MATCH: " ", ADJUSTED: "~", DIFFERS: "!"}[row["status"]]
            label = f"{row['property']:<{width}}"
            who = f" {row['variable']}" if row["variable"] else ""
            lines.append(
                f"  {mark} {label}{who:<6} requested {row['requested']!s:>12}"
                f"   got {row['achieved']!s:>12}"
            )
            if row["evidence"]:
                lines.append(f"      {row['evidence']}")
        differing = len(self.differences)
        lines.append(
            f"  {differing} of {len(self.rows)} differ from the request"
            + (
                "; ~ marks values the validator adjusted on purpose."
                if any(r["status"] == ADJUSTED for r in self.rows)
                else "."
            )
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
