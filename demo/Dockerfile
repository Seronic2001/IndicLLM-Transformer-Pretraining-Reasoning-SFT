# Railway (or any Docker host). The page runs the models in the browser; this image only serves files.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# site/models/*.onnx are left out of the upload (.railwayignore); fetch the ones missing
ARG MODELS_URL=https://github.com/Seronic2001/IndicLLM-Transformer-Pretraining-Reasoning-SFT/releases/download/demo-weights-v1
RUN python app.py --fetch-models "$MODELS_URL"
CMD ["python", "app.py"]
