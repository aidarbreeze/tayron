#!/bin/sh
# 16 проверок удобства инструментов: стенд из suspension/sculpt.js + tests/ux_tools.mjs
set -e
cd "$(dirname "$0")/.."
sh tests/mkharness.sh
exec node tests/ux_tools.mjs
