.PHONY: install lint test train clean

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	flake8 src/ tests/ --max-line-length=100

test:
	pytest -v

train:
	python src/train.py

clean:
	python -c "import os, shutil; [os.remove(os.path.join(r,f)) for r,d,fs in os.walk('.') for f in fs if f.endswith('.pyc')]; [shutil.rmtree(os.path.join(r,d), ignore_errors=True) for r,ds,fs in os.walk('.') for d in list(ds) if d in ['__pycache__','.pytest_cache']]; [os.remove(os.path.join(r,f)) for r,d,fs in os.walk('.') for f in fs if f.endswith(('.tmp','.log'))]"