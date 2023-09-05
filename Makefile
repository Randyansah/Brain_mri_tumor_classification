install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest --nbval test.py

format:
	black *.py

lint:
	pylint --disable=R,C main.py 

deploy:
	#echo "deploy goes here"
	
all: install lint test