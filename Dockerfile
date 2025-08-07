From python:3.12-slim

COPY ./requirements.txt /sciper/requirements.txt

WORKDIR /sciper

RUN pip install -r requirements.txt

COPY . /sciper

EXPOSE 8000

ENTRYPOINT ["uvicorn"]

CMD ["--host", "0.0.0.0", "FASTAPI:app"]