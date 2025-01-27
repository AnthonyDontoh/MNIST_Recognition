### Handwritten Digit Recognizer ###

This repository covers the code for performing image classification on the MNIST dataset using Deep Learning

Below are the steps to be followed:

1. Install required packages stated from requirements.txt file


For Anaconda:
Copyconda create --name <yourenvname>
conda activate <yourenvname>
pip install -r requirements.txt

For Python Interpreter:
Copy pip install -r requirements.txt

2. The entire repository is modularized into individual sections which performs specific task.

First go to src folder.
Under src folder, there are 2 primary packages:
	(a) ML_Pipeline: Contains individual modules with different function declarations to perform specific ML tasks
	(b) engine.py: this is the heart of the project, as all the function calls are done here.

3. Run/debug engine.py file and all the steps will be automatically taken care as per the logic.

4. All input datasets stored in input folder
