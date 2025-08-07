install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

lint:
	pylint --disable=R,C performance/metrics.py

freeze:
	pip freeze > requirements.txt

format:
	black *.py

deploy:
	uvicorn --host 0.0.0.0 FASTAPI:app