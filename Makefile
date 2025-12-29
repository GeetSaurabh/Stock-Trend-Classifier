PY=./venv/bin/python

.PHONY: setup ingest label features train api

setup:
	python -m venv venv
	./venv/bin/pip install -r requirements.txt

ingest:
	$(PY) src/data_ingest.py

label:
	$(PY) src/create_labels.py

features:
	$(PY) src/features/build_features.py

train:
	$(PY) src/training/train_balanced.py

api:
	./venv/bin/uvicorn api.app:app --reload
