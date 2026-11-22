import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# pytest.ini 的 --basetemp 指向 .cache/pytest-tmp，pytest 不会创建它的上级目录（新克隆的仓库里没有 .cache）
(ROOT / ".cache").mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def demo_root(tmp_path_factory):
    from adviz.demo import generate
    root = tmp_path_factory.mktemp("demo")
    generate(root, n_frames=3)
    return root


@pytest.fixture(scope="session")
def adapters():
    from adviz.adapter import load_adapters
    return load_adapters(ROOT / "adapters")


@pytest.fixture(scope="session")
def demo_ds(demo_root, adapters):
    from adviz.dataset import open_dataset
    return open_dataset(demo_root / "demo.zip", adapters=adapters)
