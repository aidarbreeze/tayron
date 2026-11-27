#!/bin/sh
# всё сразу: 20 сквозных тестов приложения + 13 проверок куба-рамки
set -e
cd "$(dirname "$0")/.."
sh tests/mkharness.sh
node tests/app20.mjs
node tests/frame_box.mjs
