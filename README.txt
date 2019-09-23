Readme for Final Project: Cats vs. Dogs
Student Name: Tsz Shing Tsoi

Package Content
----------------------------------------------------
- Readme.txt: This readme file
- final_project_starter_cap4630.m: Modified starter code for Procedures 1 to 10 of the Final Project
- final_project_final_solution_cap4630.m: Final solution for Procedures 11 to 15 of the Final Project
- imageLabelTestSetGroundTruth.csv: Manual labels of Kaggle challenge test dataset. The first column is file name and the second column is label
- FinalProject_Report_Tsoi.pdf: Final project report


Instructions to Install and Run Code
----------------------------------------------------
1. Put the .m files in this package in the same folder and set this folder as current folder in MATLAB
2. Put the imageLabelTestSetGroundTruth.csv file in the same folder, or provide your own Kaggle test data label with the same file name and format. Parts 5.1 through 5.3 of the code help prepare data for manual labeling if you would like to label the test data manually
3. Put the readAndPreprocessImage.m file provided by Dr. Marques in the same folder
4. Put the sample training data provided by Dr. Marques in folders './data/PetImages/cat' and './data/PetImages/dog'. This is required to run final_project_starter_cap4630.m
5. Put the full Kaggle training data in folder './data/train' and full Kaggle test data in folder './data/test'. These are required to run both final_project_starter_cap4630.m and final_project_final_solution_cap4630.m
6. Click Run in MATLAB and wait for completion

Notes
----------------------------------------------------
1. The random generator is re-seeded several times throughout the code for consistency and reporting purposes.
2. The code will save several intermediate output files for quicker generation of reports in sub-sequent runs. Please remove these files before you run the code as appropriate if you do not wish to re-use previous outputs:
- model*.mat: CNN model files
- model*.png: CNN model training plots
- YPred*.mat: Predictions on validation data
- ClassifierResult.mat: Predictions on test data
- imageLabelTestSetPred.csv: Labels on test data with file names for manual labelling (only if imageLabelTestSetGroundTruth.csv is not supplied)