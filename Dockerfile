FROM python:3.8-slim-buster

WORKDIR /app
COPY . .

RUN pip3 install --no-cache-dir --upgrade pip -r requirements.txt
RUN python -m nltk.downloader punkt

EXPOSE 9693
CMD [ "python3", "./app.py" ]