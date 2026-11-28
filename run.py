#!/usr/bin/env python3
"""Запуск демонстрации подвески Tayron: локальный сервер + браузер.

    python3 run.py            # сам найдёт свободный порт и откроет браузер
    python3 run.py --port 8080 --no-browser

Ничего ставить не нужно — только стандартная библиотека Python 3.8+.
"""
import argparse
import http.server
import os
import socket
import socketserver
import sys
import threading
import webbrowser

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'suspension')


class Handler(http.server.SimpleHTTPRequestHandler):
    """Тихий обработчик: без логов на каждый .glb, с корректными типами."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.js': 'text/javascript',
        '.glb': 'model/gltf-binary',
        '.json': 'application/json',
        '.wasm': 'application/wasm',
    }

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def free_port(start=8000, tries=40):
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
    raise SystemExit(f'Нет свободного порта в диапазоне {start}..{start+tries}')


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser(description='Демонстрация активной подвески Tayron')
    ap.add_argument('--port', type=int, default=0, help='порт (по умолчанию свободный с 8000)')
    ap.add_argument('--no-browser', action='store_true', help='не открывать браузер')
    ap.add_argument('--host', default='0.0.0.0', help='адрес прослушивания')
    args = ap.parse_args()

    if not os.path.isdir(ROOT):
        raise SystemExit(f'Не найдена папка с сайтом: {ROOT}')

    port = args.port or free_port()
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer((args.host, port), Handler)
    httpd.daemon_threads = True

    url = f'http://localhost:{port}/'
    ip = lan_ip()

    print()
    print('  ┌──────────────────────────────────────────────┐')
    print('  │  Tayron · активная подвеска — демонстрация   │')
    print('  └──────────────────────────────────────────────┘')
    print(f'   Открыть:      {url}')
    if ip and ip != '127.0.0.1':
        print(f'   В сети:       http://{ip}:{port}/')
    print('   Остановить:   Ctrl+C')
    print()

    if not args.no_browser:
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\n  Остановлено.')
    finally:
        httpd.server_close()


if __name__ == '__main__':
    sys.exit(main())
