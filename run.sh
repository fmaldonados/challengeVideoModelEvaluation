#!/bin/bash
# Ejecuta cargo run en el subdirectorio correcto para facilitar pruebas desde la raíz del repo
cd video_model_eval
cargo run "$@"
