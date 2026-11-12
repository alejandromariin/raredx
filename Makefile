PYTHON ?= python3

.PHONY: data split gateway gateway-stop train api eval test app app-stop

data:   ## Download and process all datasets
	$(PYTHON) -m hoofbeat.data

split:  ## Split cases by paper and pick the train/val/test samples
	$(PYTHON) -m hoofbeat.evaluate split

gateway:  ## Start only Bifrost (LLM gateway) on http://localhost:8080, to run the app or evaluation locally
	docker compose up -d bifrost

gateway-stop:
	docker compose down

train:  ## Train the fusion model on the training sample -> models/fusion.pkl
	$(PYTHON) -m hoofbeat.evaluate train-fusion

api:    ## Start the API on http://localhost:8000 (docs at /docs); needs Bifrost running
	$(PYTHON) -m uvicorn hoofbeat.api:app --port 8000

eval:   ## Run every evaluation from the caches and write results/ (no new LLM calls once cached)
	$(PYTHON) -m hoofbeat.evaluate all

test:   ## Final evaluation on the 500 test cases -> results/test.json
	$(PYTHON) -m hoofbeat.evaluate test

app:    ## Everything in Docker (Bifrost + API + interface) -> http://localhost:8000
	docker compose up -d --build
	@echo "Starting... first start downloads the data (a few minutes): docker compose logs -f app"
	@echo "Then open http://localhost:8000"

app-stop:
	docker compose down
