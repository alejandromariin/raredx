PYTHON ?= python3

.PHONY: data split gateway gateway-stop train api eval test app

data:   ## Download and process all datasets
	$(PYTHON) -m raredx.data

split:  ## Split cases by paper and pick the train/val/test samples
	$(PYTHON) -m raredx.evaluate split

gateway:  ## Start Bifrost (LLM gateway) on http://localhost:8080
	docker compose up -d bifrost

gateway-stop:
	docker compose down

train:  ## Train the fusion model on the training sample -> models/fusion.pkl
	$(PYTHON) -m raredx.evaluate train-fusion

api:    ## Start the API on http://localhost:8000 (docs at /docs); needs Bifrost running
	$(PYTHON) -m uvicorn raredx.api:app --port 8000

eval:   ## Run every evaluation from the caches and write results/ (no new LLM calls once cached)
	$(PYTHON) -m raredx.evaluate all

test:   ## Final evaluation on the 500 test cases -> results/test.json
	$(PYTHON) -m raredx.evaluate test

app:    ## Start Bifrost, backend and frontend
	@echo "TODO: docker compose up"
