FROM python:3.10.12-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -U pip
RUN pip install -r requirements.txt

ENV OPENAI_API_KEY=${OPENAI_API_KEY}

COPY . .

EXPOSE 5000

# default command
CMD ["python", "api.py", "-e", "prod"]
# CMD ["tail", "-f", "/dev/null"]
