# Classification Challenge | Module 13 Assignment 

## Background
This project was to improve ISP's email filtering system. The goal is to accurately detect and filter out spam emails from customers' inboxes using supervised machine learning techniques. The implementation compares two different classification models to determine the most effective approach for spam detection.

## Project Overview
The project implements a binary classification system that categorizes emails as either spam (1) or legitimate (0). Two different machine learning models are implemented and compared:
1. Logistic Regression: A linear classification model
2. Random Forest Classifier: An ensemble learning method

### Technical Details
- **Data Processing**: Features are scaled using StandardScaler to ensure all variables contribute equally to the model
- **Train-Test Split**: Data is split with random_state=1 for reproducibility
- **Model Parameters**: Both models use random_state=1 for consistent results
- **Evaluation Metric**: Accuracy score is used to assess model performance

## Implementation Details

### Data Preparation
1. Data loading from CSV using pandas
2. Feature extraction and label separation
3. Data split into training (75%) and testing (25%) sets
4. Feature scaling using StandardScaler
   - Fit on training data only
   - Transform both training and testing data

### Model Implementation

#### Logistic Regression
- **Implementation**: sklearn.linear_model.LogisticRegression
- **Parameters**:
  - random_state=1
  - Default solver ('lbfgs')
  - Default max_iterations (100)
- **Performance**: 92.79% accuracy on test data

#### Random Forest Classifier
- **Implementation**: sklearn.ensemble.RandomForestClassifier
- **Parameters**:
  - random_state=1
  - Default n_estimators (100 trees)
  - Default max_depth (None)
- **Performance**: 96.70% accuracy on test data

## Results Analysis

### Model Comparison
- **Logistic Regression (92.79%)**
  - Advantages: Simple, interpretable, fast training
  - Limitations: Assumes linear relationship between features
  
- **Random Forest (96.70%)**
  - Advantages: Higher accuracy, captures non-linear patterns
  - Better handling of feature interactions
  - More robust to outliers
  - Improvement of 3.91 percentage points over Logistic Regression

### Performance Analysis
The Random Forest model's superior performance can be attributed to:
1. Ability to capture complex patterns in email characteristics
2. Ensemble learning approach reducing overfitting
3. Better handling of potential outliers in the dataset
4. Effective modeling of feature interactions


## Data Source
- **Dataset**: Spambase dataset from UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/94/spambase
- **Access**: Data loaded from https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv
- **Features**: Email characteristics including word frequencies and character frequencies
- **Target Variable**: Binary classification (spam=1, legitimate=0)

## Dependencies
- Python 3.10.12
- pandas 2.2.2
- scikit-learn 1.5.1

## Installation and Usage

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/cfleming22/classification-challenge.git
   cd classification-challenge
   ```

2. Install required packages:
   ```bash
   pip install pandas==2.2.2 scikit-learn==1.5.1
   ```

### Running the Analysis
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `spam_detector.ipynb`

3. Run all cells in sequence to:
   - Load and preprocess data
   - Train both models
   - View performance comparisons

## Code Sources and Attribution
All code in this repository was independently developed following the provided starter code structure. The implementation uses standard libraries and follows established machine learning practices:

- **Original Code**:
  - All model implementation
  - Data preprocessing steps
  - Evaluation metrics calculation

- **Standard Libraries Used**:
  - pandas: Data manipulation and analysis
  - scikit-learn: Machine learning models and preprocessing

No external code sources, peer collaboration, or code sharing were involved in this project's development.

## Future Improvements
Potential enhancements for future iterations:
1. Implementation of additional models (e.g., SVM, Neural Networks)
2. Feature importance analysis
3. Hyperparameter tuning
4. Cross-validation for more robust evaluation
5. Additional evaluation metrics (precision, recall, F1-score)

## Contact
For any questions or clarifications about this implementation, please open an issue in the repository.
