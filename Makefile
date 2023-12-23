test:
	cd tests && python global-unit-test.py

lint:
	python -m pylint chefboost/ --fail-under=10