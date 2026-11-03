PYTHON ?= python3

.PHONY: data train eval app

data:   ## Download and process all datasets
	$(PYTHON) -m raredx.data

train:  ## Train the fusion model
	@echo "TODO"

eval:   ## Run the full evaluation and write results/
	@echo "TODO: $(PYTHON) -m raredx.evaluate"

app:    ## Start Bifrost, backend and frontend
	@echo "TODO: docker compose up"
