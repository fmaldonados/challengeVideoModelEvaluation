#!/bin/bash
# Ejecuta cargo test en el subdirectorio correcto para facilitar pruebas desde la raíz del repo
cd video_model_eval
cargo test "$@"
