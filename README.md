Setup on Windows 7
==================
1. Use the following instructions for setting up Anaconda for Python and Machine Learning tools including Jupyter Notebook.

https://docs.anaconda.com/anaconda/install/windows/

The distribution package to be downloaded on a Windows 64-bit system is Anaconda3-2019.10-Windows-x86_64.exe.

2. Once installed, using windows explorer, go to Anaconda prompt and run as admin as shown here
https://docs.anaconda.com/anaconda/user-guide/getting-started/#write-a-python-program-using-anaconda-prompt-or-terminal

3. Place the Propensity_Model.ipynb and nn_model.h5 in a suitable folder.

4. Invoke jupyter Notebook on the prompt navigating to the folder location as in step (3).
>> C:\Projects>jupyter notebook 

or as shown here
https://docs.anaconda.com/anaconda/user-guide/getting-started/#open-nav-win

5. Open the Propensity_Model.ipynb IPython Notebook and run all cells.

This should take a while in execution, to load all data and set up the various training, validation and test sets.

The Model training can be skipped, and instead evaluated against the test set in the last cell.
This would load the pre-trained nn_model.h5 for predictions.

To perform training, please remove the nn_model.h5 file from the IPython Notebook location so it isn't loaded.


Python Installation Package (PIP) Requirements
==============================================
The notebook contains various packages that show how to be installed from within.
Else, a pip install would be required within the anaconda environment containing Keras with Tensorflow backend in a virtual environment. 

Note: pip needs to be downloaded and installed if not already. get-pip.py is attached that can be run from a prompt where Python is available as -
>>python get-pip.py

Packages used in this project
-----------------------------
Those that require to be pip installed -
simplejson
keras
tensorflow

Those that may not be required to be installed if Anaconda or other Python environment is successfully present -
gzip
matplotlib
numpy
pandas
