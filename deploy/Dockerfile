FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV ONNX_MODEL_PATH=/app/sam2unet_conv_bghr.onnx

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app.py .
COPY sam2unet_conv_bghr.onnx .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
