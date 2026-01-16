FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime numpy requests librosa soundfile

ARG MODEL_NAME=baby_cry_classification_resnet18.onnx
ENV MODEL_NAME=${MODEL_NAME}

COPY data/splits/label_map.json data/splits/label_map.json

COPY models/${MODEL_NAME} ./${MODEL_NAME}
COPY models/${MODEL_NAME}.data ./

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]