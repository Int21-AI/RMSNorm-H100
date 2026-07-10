# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

import csv
import sys
import unittest
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = None


def load_pyproject(path):
    if tomllib is not None:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    from setuptools.config.pyprojecttoml import read_configuration

    return read_configuration(path)


class PackageMetadataTest(unittest.TestCase):
    def test_pyproject_includes_runtime_sources_and_benchmarks(self):
        config = load_pyproject("pyproject.toml")

        project = config["project"]
        setuptools = config["tool"]["setuptools"]
        package_data = setuptools["package-data"]["rmsnorm"]

        self.assertEqual(project["readme"], "README.md")
        self.assertEqual(
            project["scripts"]["rmsnorm-benchmark"],
            "rmsnorm.benchmarks.benchmark_rmsnorm:main",
        )
        self.assertIn("rmsnorm", setuptools["packages"])
        self.assertIn("rmsnorm.benchmarks", setuptools["packages"])
        self.assertNotIn("package-dir", setuptools)
        self.assertIn("csrc/*.cu", package_data)
        self.assertIn("csrc/*.cpp", package_data)

    def test_cuda_sources_do_not_depend_on_cutlass_or_cute(self):
        source_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(Path("rmsnorm/csrc").glob("*"))
            if path.suffix in {".cu", ".cuh", ".cpp", ".h"}
        ).lower()

        self.assertNotIn("#include <cutlass", source_text)
        self.assertNotIn("#include <cute", source_text)
        self.assertNotIn("#include \"cutlass", source_text)
        self.assertNotIn("#include \"cute", source_text)

    def test_current_benchmark_artifacts_match_tolerance_policy(self):
        current_dir = Path("benchmarks/results/current")
        csv_paths = sorted(current_dir.glob("*.csv"))
        self.assertGreaterEqual(len(csv_paths), 1)

        misses = []
        for csv_path in csv_paths:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    ratio_text = row.get("ratio")
                    if ratio_text in {None, "", "nan"}:
                        continue
                    ratio = float(ratio_text)
                    if ratio > 1.01:
                        misses.append((csv_path.name, row.get("M"), row.get("N"), ratio))

        self.assertEqual(misses, [])

    def test_readme_lists_current_benchmark_artifacts(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        current_files = {path.name for path in Path("benchmarks/results/current").glob("*.csv")}
        documented_patterns = [
            "autotuned-quack-backward-bfloat16-large.csv",
            "autotuned-quack-backward-bfloat16-mid.csv",
            "autotuned-quack-backward-float16-large.csv",
            "autotuned-quack-backward-float16-mid.csv",
            "autotuned-quack-forward-bfloat16-over-65536-partial.csv",
            "autotuned-quack-forward-float32-65536-row.csv",
            "autotuned-quack-forward-{bfloat16,float16,float32}-through-65536.csv",
            "autotuned-quack-residual-{bfloat16,float16}-16384-row.csv",
            "autotuned-quack-residual-{bfloat16,float16}-through-65536.csv",
            "autotuned-quack-residual-fp32-{bfloat16,float16}-16384-row.csv",
            "autotuned-quack-residual-fp32-{bfloat16,float16}-through-65536.csv",
        ]

        for pattern in documented_patterns:
            self.assertIn(f"`current/{pattern}`", readme)

        documented_files = {
            "autotuned-quack-backward-bfloat16-large.csv",
            "autotuned-quack-backward-bfloat16-mid.csv",
            "autotuned-quack-backward-float16-large.csv",
            "autotuned-quack-backward-float16-mid.csv",
            "autotuned-quack-forward-bfloat16-over-65536-partial.csv",
            "autotuned-quack-forward-float32-65536-row.csv",
        }
        for dtype in ("bfloat16", "float16", "float32"):
            documented_files.add(f"autotuned-quack-forward-{dtype}-through-65536.csv")
        for dtype in ("bfloat16", "float16"):
            documented_files.add(f"autotuned-quack-residual-{dtype}-16384-row.csv")
            documented_files.add(f"autotuned-quack-residual-{dtype}-through-65536.csv")
            documented_files.add(f"autotuned-quack-residual-fp32-{dtype}-16384-row.csv")
            documented_files.add(f"autotuned-quack-residual-fp32-{dtype}-through-65536.csv")

        self.assertEqual(current_files, documented_files)


if __name__ == "__main__":
    unittest.main()
