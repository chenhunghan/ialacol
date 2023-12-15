# syntax=docker/dockerfile:1

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip3 install -r requirements.txt
# https://github.com/marella/ctransformers#metal
RUN CT_METAL=1 pip3 install ctransformers --no-binary ctransformers
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
