#!/bin/sh
# всё сразу: 20 сквозных тестов приложения, 13 проверок куба-рамки,
# 16 проверок удобства, 17 проверок полуканата, 19 проверок прожига канала
# и студия (8 + 30 + 19 проверок файлового инструмента)
set -e
cd "$(dirname "$0")/.."
sh tests/mkharness.sh
node tests/app20.mjs
node tests/frame_box.mjs
node tests/ux_tools.mjs
node tests/rope_half.mjs
node tests/burn_cut.mjs
sh tests/run_studio.sh
