# Copyright 2026 INT21 AI
# SPDX-License-Identifier: MIT

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


if __name__ == "__main__":
    unittest.main()
