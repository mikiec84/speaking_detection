setup:
	[ -e .venv ] && virtualenv --python=python3.6 .venv
	.venv/bin/pip install poetry
	.venv/bin/poetry install -vvv
i:
	.venv/bin/ipython
n:
	.venv/bin/jupyter notebook .
