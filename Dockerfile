FROM public.ecr.aws/lambda/python:3.12

ENV NUMBA_DISABLE_JIT=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN pip install --no-cache-dir \
    onnxruntime numpy requests soundfile scipy

ARG MODEL_NAME=baby_cry_classification_resnet18.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY data/splits/label_map.json data/splits/label_map.json

COPY models/onnx/${MODEL_NAME} ./${MODEL_NAME}
COPY models/onnx/${MODEL_NAME}.data ./

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]