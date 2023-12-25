test:
	cd tests && python -m pytest . -s --disable-warnings

lint:
	python -m pylint chefboost/ --fail-under=10