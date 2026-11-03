PYTHON ?= python3

.PHONY: data split gateway gateway-stop train eval app

data:   ## Download and process all datasets
	$(PYTHON) -m raredx.data

split:  ## Split cases by paper and pick the train/val/test samples
	$(PYTHON) -m raredx.evaluate split

gateway:  ## Start Bifrost (LLM gateway) on http://localhost:8080
	docker compose up -d bifrost

gateway-stop:
	docker compose down

train:  ## Train the fusion model
	@echo "TODO"

eval:   ## Run the full evaluation and write results/
	@echo "TODO: $(PYTHON) -m raredx.evaluate"

app:    ## Start Bifrost, backend and frontend
	@echo "TODO: docker compose up"
