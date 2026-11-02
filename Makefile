.PHONY: data train eval app test

data:   ## Download and process all datasets
	@echo "TODO: python -m raredx.data"

train:  ## Train the fusion model
	@echo "TODO"

eval:   ## Run the full evaluation and write eval/results/
	@echo "TODO: python -m eval.run_eval"

app:    ## Start Bifrost, backend and frontend
	@echo "TODO: docker compose up"

test:
	pytest
