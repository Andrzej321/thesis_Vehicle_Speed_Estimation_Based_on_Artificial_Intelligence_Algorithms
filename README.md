This project was created for my thesis. The title is Vehicle Speed Estimation Based on Artificial Intelligence Algorithms.

The main parts of the project are the following:
1. matlab
    1. Data extraction
        - The extraction and the preprocessing of the measurement data is done.
    2. AI training in maltab
        
        !!! These scripts use the Deep Learning Toolbox from matlab !!!
        
        - These training sequences are created to be able to generate c code from the ai models. This was done due to not being able to import and generate c code from the onnx models that were trained with PyTorch. 
        - three models can be trained here: GRU, LSTM, TCN
    3. Validation  
        - The created models and conventional estimators are simulated here and validated in the estimator_designs_main.slx Simulink model.
        - The validated estimators are: Arbitrary ai model (two ways), best wheel based (bw), model based with linear tire model (m), ekf with m as the state update transition function and bw as the measurement update function, ekf with m as the state update transition function and ai as teh measurement update funcitno. 
            - As for the arbitrary ai models. There are two ways to use these models. The first uses a predict function from the Deep Learning Toolbox. The second utilizes a wrapper function that was done by Balázs. The second option was done to speed up the testing phase.
            - The first method is used in the latter real-time testing that was done on OTB, due to the c code generation.  
2. python
    - This folder contains the scripts that are used for training the ai models in python. The main scripts are the following: training_RNN.py, training_LSTM.py, training_GRU.py, training_TCN.py, training_T.py. The model that the script trains is indicated in the name of the files. The principle is the same in these training scripts which is the following: defining the paths to the folders containing the training and validation measurement files, defining the paths to save the trained models, providing the path to the csv file that contains the hyperparameters of the specific models. These csv files are in the folder of the trained models.
    - Additional validation and evaluation scripts are also present. The validation script imports the checkpoints of the saved models that are within a defined folder and performs a regression based on the inputs from the testing measurement files. The output of these script is a set of result files that contain all the outputs of the models that were in the folder. Next the evaluation script imports these model specific result files and performs validation file specific and summarized evalutaion files based on the root mean square error (RMSE), mean absolute error (MAE), maximum absolute error (MaxAE) and a percantage value that indicates the points within a certain error threshold. 
3. data
    - This folder contains all the measurement data that was used for training and testing purposes. The source of the used files are the measurement in the folder 1_raw_meas_files/Tokol_2024_11_15_VSE+DSC folder. The desired CAN data is extracted from these measurements. After postprocessing these measurement files, the already usable csv and mat files can be found in the 2_extracted_meas_files folder. The variants in this folder indicate the sampling rate and the normalization of the measurement data.
4. trained models
    - This folder contains all the files that regarding the trained models and the training data that is distributed to training, validation and testing datasets. Every folder stroing the trained models has the same structure. There are iterations where different architectures where trained, tested and evaluated. The csv file containing the hyperparameters is in these iteration subfolders.
5. results
    - The results folder contains the best performing models, the results of the conventional models created in Matlab Simulink (1. folder) and the result file containing all the results of the mentioned estimators. 
