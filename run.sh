#!/bin/bash

echo "Starting..."

python3 src/web_streaming_classify.py \
        --model models/inception_v2_224_quant_edgetpu.tflite \
        --label models/imagenet_labels.txt

sleep infinity
