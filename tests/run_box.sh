#!/bin/sh
# 13 проверок куба-рамки: стенд из suspension/sculpt.js + tests/frame_box.mjs
set -e
cd "$(dirname "$0")/.."
sh tests/mkharness.sh
exec node tests/frame_box.mjs
