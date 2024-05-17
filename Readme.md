# Phone Price Prediction


Requirements

    python 3.10.10
    numpy==1.24.3
    pandas==2.0.1
    scikit-learn==1.0

Running:

    To run the demo, execute:
        python predict.py 

    After running the script in that folder will be generated <predictions.csv> 

    The input is expected  csv file in the same folder with a name <new_data.csv>. The file should have all features columns. 

Training a Model:

    Before you run the training script for the first time, you will need to create dataset. The file <train.csv> should contain all features columns and target for prediction Exited.
    After running the script the "dtree_model.saw" will be created.
    Run the training script:
        python train.py
