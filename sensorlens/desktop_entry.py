"""桌面版后端入口。

sensorlens-backend serve --port 8765 --data <数据目录>   启动服务（桌面外壳调用）
sensorlens-backend cli render <输入> --out <目录>        批量命令行，同 python -m sensorlens.cli
"""
from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import threading
from pathlib import Path


def _resources() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def _watch_stdin() -> None:
    """外壳通过管道持有本进程的 stdin；外壳退出（包括崩溃）时管道关闭，这里随之退出，不留孤儿进程。

    Windows 上同步管道的阻塞读会让同一句柄上的其他调用（导入时的 isatty/fstat）一起卡住，
    所以复制出一个句柄专门给监视线程读，fd 0 换成空设备。
    """
    fd = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    sys.stdin = open(os.devnull, encoding="utf-8")

    def run() -> None:
        try:
            while os.read(fd, 1024):
                pass
        except OSError:
            pass
        os._exit(0)

    threading.Thread(target=run, daemon=True).start()


def serve(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="sensorlens-backend serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", required=True, help="数据目录（会话、缓存、日志）")
    ap.add_argument("--watch-stdin", action="store_true", help="stdin 关闭时退出")
    a = ap.parse_args(argv)
    data = Path(a.data).resolve()
    data.mkdir(parents=True, exist_ok=True)
    os.environ["SENSORLENS_DATA"] = str(data)
    os.environ.setdefault("SENSORLENS_RES", str(_resources()))
    tmp = data / "tmp"
    tmp.mkdir(exist_ok=True)
    for k in ("TEMP", "TMP", "TMPDIR"):
        os.environ[k] = str(tmp)
    if a.watch_stdin:
        _watch_stdin()
    import uvicorn

    from sensorlens.server import app
    print(f"SensorLens backend on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", access_log=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "cli":
        from sensorlens.cli import main as cli_main
        return cli_main(argv[1:])
    if argv and argv[0] == "serve":
        argv = argv[1:]
    return serve(argv)


if __name__ == "__main__":
    sys.exit(main())
