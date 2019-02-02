# Art_Classification-Virtual_Museum

# Instructions to Install:

## For Mac users:

pip install tensorflow

pip install keras

conda update --all

## For Windows users:

Open Anaconda Prompt:

1. Create a new environment with Anaconda and Python 3.5:
	
	conda create -n myEnv python=3.5 anaconda

2. Activate the environment:
	
	activate myEnv

3. install TensorFlow and Keras:
	
	conda install mingw libpython
	pip install tensorflow
	pip install keras
	
	conda update --all
# Tools:

Anaconda - Jupyter notebook:

Anaconda IDE can be downloaded from the below mentioned url:
https://www.anaconda.com/download

You can download the respective version depending on the type of operating system.
The python version used for this project is 3.6.
You can open the anaconda and install the jupyter notebook.

# Libraries:

Commands to install various libraries are mentioned below::

1. Tensorflow: pip install tensorflow
2. Keras: pip install keras
3. XGBoost: pip install xgboost
4. matplotlib : pip install matplotlib
5. scikit-learn : pip install scikitlearn
6. numpy
7. pandas

# Instructions to Run Code Snippets

## CNN
- Open CNN_Training.ipynb from Source_Code/CNN
- Edit **traning_dir** and **test_dir** variables as per location of the training and validation folder from the dataset in your local machine.
- Run cells. 

## Pre-Tranied models 
Open Source_Code/PreTrained_Models
Five Files :
- Pretrained_DenseNet201 
- Pretrained_InceptionRenNetV2
- Pretrained_ResNet50
- Pretrained_VGG16
- Pretrained_Xception

- To run any of the above files, make changes in **training_dir** and **test_dir** variables as per location of the training and validation folder from the dataset in your local machine. 

## Feature_Extraction 

- Open Source_Code/Transfer_Learning/Feature_Extraction.
- The folder contained four files.
	- Feature_Extraction_Train_resnet.ipynb
	- Feature_Extraction_Train_vgg.ipynb
	- Feature_Extraction_Test_resnet.ipynb 
	- Feature_Extraction_Test_vgg.ipynb
- To run any of the files, make below mentioned changes 
	- make **path** varible point to training set or validation set based on the script you're trying tp run. 
		**example: for Feature_Extraction_Train_resnet - path= "/training_set" and for Feature_Extraction_Test_resnet - path="/validation_set ** 
	- change df.to_csv("give path to store X_train or X_test based on the script you are trying to run on your local machine") **example : csvfile= "/X_test_resnet.csv"**   
	- change **csvfile** path variable that points to store Y_train or Y_test.csv based on the script you are trying to in your local machine. **example : csvfile= "/Y_test_resnet.csv"**  
## Classification

- Open Source_Code/Transfer_Learning 
- The folder conatins following files and folder:
	- Feature Extraction Folder
	- Classification_resnet.ipynb 
	- Classification_vgg.ipynb
	
- **Prerequisites** : Classification_resnet.ipynb or Classification_vgg.ipynb, make sure you already have features extracted files like X_train_vgg.csv,Y_train_vgg.csv,X_test_vgg.csv,Y_test_vgg.csv, X_train_resnet.csv, Y_train_resnet.csv, X_test_resnet.csv and Y_test_resnet.csv  or  get files using previously mentioned Feature Extraction scripts. 

- To run any of the two scripts, change below mentioned variables:
	- xTrain- path to X_train.csv on your local machine 
	- yTrain - path to y_train.csv on your local machine
	- xTest - path to X_test.csv on your local machine
	- yTest - path to y_test.csv on your local machine



