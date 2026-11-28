#!/bin/sh
# 20 сквозных тестов лепки: стенд из suspension/sculpt.js + tests/app20.mjs
set -e
cd "$(dirname "$0")/.."
sh tests/mkharness.sh
exec node tests/app20.mjs
