#!/bin/bash
cd ~/llama.cpp/build/bin
./llama-server \
  --model /storage/emulated/0/Download/gpt-oss-20b-mxfp4.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --threads 6 \
  --ctx-size 4096 \
  --n-gpu-layers 50
