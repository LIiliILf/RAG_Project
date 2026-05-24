"""
第13节：Gradio 应用启动脚本。

用途：
1. 提供统一启动入口。
2. 通过参数控制监听地址和端口。
"""

from pathlib import Path
import argparse
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from gradio_app import launch_gradio_app


def parse_args():
    parser = argparse.ArgumentParser(description="启动第13节 Gradio 界面。")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址。")
    parser.add_argument("--port", type=int, default=7860, help="监听端口。")
    parser.add_argument("--share", action="store_true", help="是否启用 share 链接。")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"启动 Gradio 应用: http://{args.host}:{args.port}")
    launch_gradio_app(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

