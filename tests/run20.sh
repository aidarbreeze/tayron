#!/bin/sh
# 20 сквозных тестов лепки: собирает стенд из suspension/sculpt.js и гоняет tests/app20.mjs
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -f tests/body.json ] || python3 tests/mkbody.py
mkdir -p tests/.harness
python3 - "$ROOT" <<'PY'
import sys
root = sys.argv[1]
s = open(root + '/suspension/sculpt.js', encoding='utf-8').read()
s = s.replace("from 'three'", "from '%s/suspension/vendor/three.module.js'" % root)
for m in ('export3mf', 'meshops', 'ropes'):
    s = s.replace("from './%s.js'" % m, "from '%s/suspension/%s.js'" % (root, m))
open(root + '/tests/.harness/sculpt_real.mjs', 'w', encoding='utf-8').write(s)
PY
exec node tests/app20.mjs
