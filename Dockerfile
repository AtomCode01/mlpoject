FROM python:3.10-slim-buster
WORKDIR /app
COPY ./ app
RUN apt update -y && apt install awscli -y

RUN pip3 install --no-cache-dir -r requirements.txt
CMD ["python3", "app.py"]
