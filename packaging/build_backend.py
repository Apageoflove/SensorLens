# 构建 Python 后端（PyInstaller onedir），输出到 build/backend，供 Tauri 作为资源打包。
# 用法：python packaging/build_backend.py
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
dist, work = root / "build" / "pyi" / "dist", root / "build" / "pyi" / "work"
subprocess.check_call([sys.executable, "-m", "PyInstaller", str(root / "packaging" / "sensorlens-backend.spec"),
                       "--noconfirm", "--clean", "--distpath", str(dist), "--workpath", str(work)], cwd=root)
out = root / "build" / "backend"
if out.exists():
    shutil.rmtree(out)
shutil.copytree(dist / "sensorlens-backend", out)
exe = out / ("sensorlens-backend.exe" if sys.platform == "win32" else "sensorlens-backend")
if not exe.is_file():
    sys.exit(f"没有生成 {exe}")
print(f"backend -> {out}")
