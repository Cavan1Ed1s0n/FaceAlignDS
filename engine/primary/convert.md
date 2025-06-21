/usr/src/tensorrt/bin/trtexec --onnx=./yolov8n-face.onnx \
        --saveEngine=./yolov8n-face_b1_fp16.engine \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:1x3x640x640 \
        --maxShapes=input:1x3x640x640 \
        --verbose

/usr/src/tensorrt/bin/trtexec --onnx=./yolov8n-face.onnx \
        --saveEngine=./yolov8n-face_b10_fp16.engine \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:10x3x640x640 \
        --maxShapes=input:10x3x640x640 \
        --verbose

/usr/src/tensorrt/bin/trtexec --onnx=./yolov8n-face-new.onnx \
        --saveEngine=./yolov8n-face-new_b1_fp16.engine \
        --minShapes=input:1x3x640x640 \
        --optShapes=input:1x3x640x640 \
        --maxShapes=input:1x3x640x640 \
        --verbose