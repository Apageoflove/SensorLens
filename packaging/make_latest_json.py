"""根据 CI 产物生成 tauri-plugin-updater 读取的 latest.json。

用法：python packaging/make_latest_json.py <产物目录> <版本号> <下载地址前缀> [说明]
产物目录里找安装包和同名 .sig；只收录有签名的安装包。
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

art, version, base = Path(sys.argv[1]), sys.argv[2], sys.argv[3].rstrip("/")
notes = sys.argv[4] if len(sys.argv) > 4 else f"SensorLens v{version}"

# 更新器先找 {os}-{arch}-{安装方式}，找不到再用 {os}-{arch}
RULES = [
    ("-setup.exe", ["windows-x86_64-nsis", "windows-x86_64"]),
    (".AppImage", ["linux-x86_64-appimage", "linux-x86_64"]),
    (".deb", ["linux-x86_64-deb"]),
    (".rpm", ["linux-x86_64-rpm"]),
]
platforms = {}
for sig in sorted(art.rglob("*.sig")):
    pkg = sig.with_suffix("")
    if not pkg.is_file():
        continue
    for suffix, keys in RULES:
        if pkg.name.endswith(suffix):
            entry = {"signature": sig.read_text(encoding="utf-8").strip(), "url": f"{base}/{pkg.name}"}
            for k in keys:
                platforms.setdefault(k, entry)
for need in ("windows-x86_64", "linux-x86_64"):
    if need not in platforms:
        sys.exit(f"缺少 {need} 的已签名安装包")
doc = {"version": version, "notes": notes, "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "platforms": platforms}
(art / "latest.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v["url"] for k, v in platforms.items()}, indent=2))
