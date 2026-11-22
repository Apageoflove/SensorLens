# PyInstaller 配置：把 Python 后端打成一个目录（onedir），由桌面外壳作为资源一起安装。
# 用法（在仓库根目录）：pyinstaller packaging/adviz-backend.spec --noconfirm --distpath build/pyi/dist --workpath build/pyi/work
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

root = Path(SPECPATH).parent
datas = [(str(root / d), d) for d in ("web", "adapters", "config", "samples")]
hidden = collect_submodules("uvicorn") + ["python_multipart", "multipart"]

a = Analysis([str(root / "packaging" / "backend_main.py")], pathex=[str(root)], datas=datas, hiddenimports=hidden,
             excludes=["tkinter", "matplotlib", "scipy", "pytest", "IPython", "PyQt5", "PyQt6", "PySide6"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="adviz-backend", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="adviz-backend", upx=False)
