# DAEN690 - Team Apache - Rotorcraft Phase Identification

This project helps identify flight phases of rotorcraft by developing a physics model following the research paper [Phases of Flight Identification for Rotorcraft Operations
](https://www.researchgate.net/publication/330196863_Phases_of_Flight_Identification_for_Rotorcraft_Operations) to label the data and employ ML techniques to help develop a model that can automatically identify the phases.

## Dependencies

```
pandas
numpy
matplotlib
mplcursors
scikit-learn
seaborn
imblearn
flaml
plotly
tensorflow
```

## Install
```
python3 -m venv apache-env
source apache-env/bin/activate
pip3 install -r requirements.txt

See individual notebooks for specific installation instructions
```

## Project structure
```
├── models
│   ├── auto_ml.ipynb
│   ├── auto_ml_sw_rc.ipynb
│   ├── lstm.ipynb
│   └── logistic_regression_decisiontree_and_NN.ipynb
├── data_quality_check.ipynb
├── data_analysis.ipynb
├── data_preprogressing_physics_model.ipynb
├── data_visualization.ipynb
├── best_model.pkl
├── requirements.txt
└── README.md
```

## Run the best model
Load the best model and make a prediction:
```
import pickle
import flaml
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the best model for prediction
with open("best_model.pkl", "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Load the flight data
data = pd.read_csv('flight_data.csv')
X = data[['Vert. Speed', 'Groundspeed', 'Altitude(AGL)']]

# Make predictions using the loaded model
phases = loaded_model.predict(X)
```

## Notebooks

- **data_analysis.ipynb**
After the simulator data was received, in this notebook, we did an initial analysis exploring the different columns, plotting the correlation matrix between the columns, and doing unsupervised learning trying to identify how many flight phases in the data by clustering the data and finding the best number of clusters using the elbow method.
- **data_quality_check.ipynb**
This notebook has some of the data quality checks that we have performed to check the dataset completeness, consistency, and uniqueness. 
- **data_preprogressing_physics_model.ipynb**
After identifying the issues in the dataset, we started pre-progressing it to address them, removing duplicate data and empty values and formatting all the columns to the right data type. We have also developed a physics model to label the rows. The result of this notebook is a labeled dataset ready for use in Machine Learning.
- **data_visualization.ipynb**
To help understand the identified phases during a flight and verify the physics model's accuracy, we have visualized the data into multiple graphs showing the identified phases at different altitudes. Also, we have plotted the identified phases versus the groundspeed and the vertical speed. To help show the different phases at different GPS coordinates, we have created a graph showing the identified phases along with the longitude and latitude coordinates. Finally, to help understand how many phases occurred within a flight, we extracted one single flight and used a roll-up sliding window technique to show the count of each phase in a single flight.

### Models

- **auto_ml.ipynb**
Employing AutoML using the Flaml python open-source library, we have loaded the labeled dataset to identify the importance of each feature and validate the physics model features. After that, we have developed the model training to find the best model.
- **auto_ml_sw_rc.ipynb**
Employing AutoML and sliding window regression classifier to find the count of each identified phase.
- **auto_ml_feature_importance.ipynb**
Employing AutoML to identify which features have the most impact on the outcome of the model. To improve the final model accuracy and validate the physical model performance.
- **lstm.ipynb**
Training LSTM (Long Short-Term Memory) a recurrent neural network (RNN) to identify the flight phases.
- **logistic_regression_decisiontree_and_NN.ipynb**
Training Logistic Regression, Decision Tree, and Neural Network.

### Final Results

| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Auto ML (RandomForest) | 1.00     | 1.00      | 1.00   | 1.00     |
| Neural Network         | 0.92     | -         | -      | -        |
| Logistic Regression    | 0.87     | 0.87      | 0.87   | 0.86     |
| Decision Tree          | 0.95     | 0.96      | 0.95   | 0.95     |
| LSTM                   | 0.94     | 0.91      | 0.94   | 0.92     |
