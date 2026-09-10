"""C01 冒烟：可编辑安装后包可导入、子包齐全。"""

import calibinfo

SUBPACKAGES = [
    "calibinfo.models",
    "calibinfo.information",
    "calibinfo.estimators",
    "calibinfo.datasets",
    "calibinfo.metrics",
    "calibinfo.io",
]


def test_calibinfo_import():
    assert calibinfo.__version__ == "0.2.0"


def test_subpackages_present():
    import importlib

    for name in SUBPACKAGES:
        importlib.import_module(name)
