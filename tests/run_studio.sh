#!/bin/sh
# Tayron Studio: самопроверка разборщиков, API через настоящий HTTP и страница
# редактора в node (настоящие three + sculpt.js + studio.js, фальшивый только DOM)
set -e
cd "$(dirname "$0")/.."

echo "— самопроверка studio.py (разбор GLB/STL/3MF, запись, хранилище)"
python3 studio.py --self-test

echo
echo "— лаунчер: скрипт страницы синтаксически цел"
python3 - <<'PY'
import re, subprocess, sys, tempfile, os
import studio
import tempfile
html = studio.launcher_html(studio.Store(tempfile.mkdtemp(prefix='studio_lch_')))
scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
assert scripts, 'в лаунчере нет скрипта'
with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
    f.write('\n'.join(scripts))
    name = f.name
r = subprocess.run(['node', '--check', name], capture_output=True, text=True)
os.unlink(name)
if r.returncode:
    print(r.stderr[:800]); sys.exit(1)
print(f'  ok   {len(html)} байт, {sum(s.count(chr(10)) for s in scripts)} строк скрипта — синтаксис в порядке')
PY

echo
echo "— API студии через настоящий HTTP"
python3 tests/studio_api.py

echo
echo "— страница редактора: загрузка, правка мышью, сохранение версий"
node tests/studio_page.mjs
