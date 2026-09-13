"""C04 绑定测试 · run manifest。

验收（主控计划书04）：干跑一次生成 manifest.json 字段齐全；
config hash 对内容差异敏感、对键序/空白不敏感。
"""

import json

from calibinfo.io.manifest import config_hash, run_manifest, write_manifest

FIELDS = ["experiment", "git_sha", "git_dirty", "python", "numpy", "scipy",
          "platform", "hostname", "config", "config_hash", "seed",
          "dataset_checksums", "started_utc", "written_utc"]


def test_manifest_fields_complete(tmp_path):
    out = tmp_path / "raw" / "ci00_dry" / "manifest.json"
    mf = write_manifest(out, config={"a": 1}, experiment="ci00_dry", seed=7,
                        repo_root=".")
    text = json.loads(out.read_text(encoding="utf-8"))
    for k in FIELDS:
        assert k in mf and k in text


def test_config_hash_sensitivity():
    a = config_hash({"m": 40, "q": 3, "seed": 7})
    assert a == config_hash({"seed": 7, "q": 3, "m": 40})          # 键序不敏感
    assert config_hash({"m": 40, "q": 3, "seed": 8}) != a          # 内容敏感
    assert config_hash({"m": 40, "q": 4, "seed": 7}) != a
    assert config_hash({"m": 40, "q": 3, "seed": 7, "extra": ""}) != a


def test_config_hash_default_repr_not_str():
    """T9 · canonical 序列化必须用 repr 而非 str:T9 前 default=str 会把
    __str__ 相同但 __repr__ 不同的对象塌成同一 hash。"""

    class _Collider:
        def __init__(self, val):
            self.val = val

        def __str__(self):
            return "<collider>"          # 故意让 str 相同

        def __repr__(self):
            return f"_Collider({self.val!r})"

    h1 = config_hash({"x": _Collider(1)})
    h2 = config_hash({"x": _Collider(2)})
    assert h1 != h2                     # 值不同 ⇒ repr 不同 ⇒ hash 不同


def test_git_sha_present_in_repo():
    from pathlib import Path
    repo_root = str(Path(__file__).resolve().parents[1])
    mf = run_manifest(config={}, repo_root=repo_root)
    assert mf["git_sha"] and len(mf["git_sha"]) == 40
