# Introduction to Artificial Intelligence Final Project  
## Cats vs. Dogs

### File Descriptions
- **CAP4630_FinalProject_Summer2019.pdf**: Project details and scopes  
- **FinalProject_Report_Tsoi.pdf**: Final project report  
- **final_project_starter_cap4630.m**: MATLAB codes for Procedures 1 to 10 of the project, modified from the start code provided by professor  
- **final_project_final_solution_cap4630.m**: MATLAB codes for Procedures 11 to 15 of the project  
- **readAndPreprocessImage.m**: MATLAB codes for function `readAndPreprocessImage()` provided by professor  
- **imageLabelTestSetGroundTruth.csv**: Manual labels of Kaggle challenge test dataset. The first column is file name and the second column is label  
- **sample cats vs dogs data.zip**: Sample training data provided by professor  

----------------------------------------------------

### Instructions to Install and Run Code
1. Put the .m files in this repository in the same folder and set this folder as current folder in MATLAB
2. Put the imageLabelTestSetGroundTruth.csv file in the same folder, or provide your own Kaggle test data label with the same file name and format. Parts 5.1 through 5.3 of the code help prepare data for manual labeling if you would like to label the test data manually
3. Put the sample training data provided by professor in folders './data/PetImages/cat' and './data/PetImages/dog'. This is required to run final_project_starter_cap4630.m
4. Put the full Kaggle training data (downloadable from https://www.kaggle.com/c/dogs-vs-cats/data) in folder './data/train' and full Kaggle test data in folder './data/test'. These are required to run both final_project_starter_cap4630.m and final_project_final_solution_cap4630.m
5. Click Run in MATLAB and wait for completion

----------------------------------------------------

### Notes
1. The random generator is re-seeded several times throughout the code for consistency and reporting purposes.
2. The code will save several intermediate output files for quicker report generation in sub-sequent runs. Please remove these files before you run the code as appropriate if you do not wish to re-use previous outputs:
- model*.mat: CNN model files
- model*.png: CNN model training plots
- YPred*.mat: Predictions on validation data
- ClassifierResult.mat: Predictions on test data
- imageLabelTestSetPred.csv: Labels on test data with file names for manual labelling (only if imageLabelTestSetGroundTruth.csv is not supplied)
