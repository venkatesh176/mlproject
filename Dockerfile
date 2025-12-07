FROM python:3.13-slim-bullseye
WORKDIR /app
COPY . /app

RUN apt-get update -y && \
    apt-get install -y awscli && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt
CMD ["python", "app.py"]
