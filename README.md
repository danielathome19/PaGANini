# About
PaGANini is a generative adversarial network trained to compose virtuosic classical music with an SVM discriminator system (C-RNN-GAN) 
To find out more, check out the provided research paper:
  * "Generative Deep Learning for Virtuosic Classical Music.pdf" (arXiv:2101.00169)

The survey mentioned in my research paper for the musical likeness can be found at [GanSurvey](http://danielszelogowski.com/gansurvey/).

# Usage
For data used in my experiments:
  * All datasets can be found in [AllDatasets.zip](https://yadi.sk/d/BdyBP6SpI1-HVg), only the full original dataset (the "paganini" folder) is provided in this repository.  
  * My most recent pre-trained weights can be found in [Weights.zip](https://yadi.sk/d/CULFtvqj9nMbjA).
  * All output compositions by my model can be found in [ProjectOutput.zip](https://yadi.sk/d/-J_1BUkPikiedw).

**NOTE:** these folders should be placed in the **same** folder as "finalProject.py". For folder existing conflicts, simply merge the directories.


In finalProject.py, the "fpmain" function acts as the controller for the model, where calls to train the model, create a prediction, run the SVM, and all other functions are called.
The code has been updated to include a call to fpmain at the bottom, but one may also call this function from an external script ("from finalProject import fpmain").

To choose an operation or series of operations for the model to perform, simply edit the fpmain function before running. Examples of all function calls can be seen commented out within fpmain.

# Bugs/Features
Bugs are tracked using the GitHub Issue Tracker.

Please use the issue tracker for the following purpose:
  * To raise a bug request; do include specific details and label it appropriately.
  * To suggest any improvements in existing features.
  * To suggest new features or structures or applications.
