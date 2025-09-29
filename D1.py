import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV,
    StratifiedKFold,
    validation_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings('ignore')

# File path for local file
FILEPATH = 'breast+cancer+wisconsin+diagnostic/wdbc.data'

class BreastCancerClassifier:
    """
    Breast Cancer Classification using Scikit-learn best practices.

    Features:
    - Scikit-learn Pipeline for data preprocessing
    - GridSearchCV for hyperparameter optimization
    - Cross-validation with StratifiedKFold
    - Comprehensive model evaluation
    - Feature selection capabilities
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Scikit-learn components
        self.pipeline = None
        self.best_model = None
        self.feature_names = None
        self.class_names = None

        # Results storage
        self.results = {}


    def load_data(self, filepath = FILEPATH):
        # Load breast cancer dataset from .data file or give error
        print(" LOADING BREAST CANCER DATASET")
        print("-" * 50)

        if filepath:
            try:
                self.data = pd.read_csv(filepath, header=None)
                print(f"Successfully loaded {len(self.data)} samples from {filepath}")
            except FileNotFoundError:
                print(f"File not found: {filepath}")
            except Exception as e:
                print(f"Error loading file: {e}")
        else:
            print("No file provided. ")
            exit()

        self._process_data()
        return self.data

    def _process_data(self):
        # Process the loaded data using scikit-learn best practices
        print(f"\nPROCESSING DATA WITH SCIKIT-LEARN")

        # Extract components: ID, Diagnosis, Features
        ids = self.data.iloc[:, 0]
        diagnosis = self.data.iloc[:, 1]
        features = self.data.iloc[:, 2:].astype(float)

        # Simple binary encoding: M=1 (Malignant), B=0 (Benign)
        self.y = (diagnosis == 'M').astype(int)
        self.X = features
        self.class_names = ['Benign', 'Malignant']  # 0=Benign, 1=Malignant

        # Create feature names
        self.feature_names = [f'feature_{i:02d}' for i in range(len(self.X.columns))]
        self.X.columns = self.feature_names

        # Display information
        print(f"Data processed successfully:")
        print(f"Total samples: {len(self.data)}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {self.class_names}")
        print(f"Class distribution:")
        for i, class_name in enumerate(self.class_names):
            count = sum(self.y == i)
            print(f"      {class_name} ({i}): {count} ({count / len(self.y) * 100:.1f}%)")

    def create_pipeline(self, use_feature_selection=True, n_features=20):
       #Create scikit-learn pipeline for preprocessing and modeling.
        print(f"\n  SCIKIT-LEARN PIPELINE")

        steps = []
        # Feature selection (optional)
        if use_feature_selection:
            steps.append(('feature_selection', SelectKBest(f_classif, k=n_features)))
            print(f" Feature selection: Top {n_features} features")

        # Scaling
        steps.append(('scaler', StandardScaler()))
        print(f" Feature scaling: StandardScaler")

        # Classifier
        steps.append(('classifier', LogisticRegression(random_state=self.random_state, max_iter=2000)))
        print(f" Classifier: Logistic Regression")

        # Create pipeline
        self.pipeline = Pipeline(steps)
        print(f" Pipeline created with {len(steps)} steps")

        return self.pipeline

    def split_data(self, test_size=0.2, stratify=True):
        #Split data using scikit-learn train_test_split.
        print(f"\n SPLITTING DATA")

        stratify_param = self.y if stratify else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        print(f" Data split completed:")
        print(f" Training: {len(self.X_train)} samples ({(1 - test_size) * 100:.0f}%)")
        print(f" Testing: {len(self.X_test)} samples ({test_size * 100:.0f}%)")
        print(f" Stratified: {'Yes' if stratify else 'No'}")

        # Show class distribution in splits
        print(f" Training distribution: {np.bincount(self.y_train)}")
        print(f" Testing distribution: {np.bincount(self.y_test)}")

    def optimize_hyperparameters(self):
       #Use GridSearchCV for hyperparameter optimization.
        print(f"\n HYPERPARAMETER OPTIMIZATION")
        print("-" * 35)

        # Define parameter grid
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga'],
            'classifier__max_iter': [2000]  # Fixed max_iter to prevent convergence warnings
        }

        # Create GridSearchCV with StratifiedKFold
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv_strategy,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        print(f" Running GridSearchCV...")
        print(f" Parameters to optimize: {list(param_grid.keys())}")
        print(f" Cross-validation folds: 5")
        print(f" Scoring metric: ROC-AUC")

        # Fit the grid search
        grid_search.fit(self.X_train, self.y_train)

        # Store best model
        self.best_model = grid_search.best_estimator_

        print(f"Optimization completed:")
        print(f"Best ROC-AUC: {grid_search.best_score_:.4f}")
        print(f"Best parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"      {param}: {value}")

        return grid_search

    def evaluate_model(self):
        #Comprehensive model evaluation using scikit-learn metrics.
        print(f"\nMODEL EVALUATION")

        # Predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # Basic metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)

        print(f"PERFORMANCE METRICS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        print(f"   F1-Score:  {f1:.4f}")

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.best_model, self.X_train, self.y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='accuracy'
        )

        print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\n CONFUSION MATRIX:")
        print(f"                 Predicted")
        print(f"                 {self.class_names[0]:^6} {self.class_names[1]:^6}")
        print(f"Actual {self.class_names[0]:>6}   {cm[0, 0]:^6d} {cm[0, 1]:^6d}")
        print(f"       {self.class_names[1]:>6}   {cm[1, 0]:^6d} {cm[1, 1]:^6d}")

        # Detailed metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"\n DETAILED METRICS:")
        print(f"   Sensitivity (Recall): {sensitivity:.4f}")
        print(f"   Specificity:          {specificity:.4f}")
        print(f"   Precision:            {precision:.4f}")

        # Classification Report
        print(f"\n CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_pred, target_names=self.class_names))

        # Dataframe for visualization
        metrics_df = pd.DataFrame({
            'Metric': ['Sensitivity (Recall)', 'Specificity', 'Precision'],
            'Value': [sensitivity, specificity, precision]
        })

        # Draw diagram
        plt.figure(figsize=(8, 6))
        plt.bar(metrics_df['Metric'], metrics_df['Value'], color='skyblue')
        plt.ylim(0, 1.1)
        plt.title('Detailed Classification Metrics')
        plt.ylabel('Score')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Feature importance (if available)
        self._show_feature_importance()

        # Store results
        self.results = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }



        return self.results

    def _show_feature_importance(self):
        #Display feature importance from the trained model.
        try:
            # Get the classifier from the pipeline
            classifier = self.best_model.named_steps['classifier']

            if hasattr(classifier, 'coef_'):
                print(f"\n TOP 10 FEATURE IMPORTANCE (Coefficients):")
                print("-" * 50)

                # Get feature names after selection (if used)
                if 'feature_selection' in self.best_model.named_steps:
                    selected_features = self.best_model.named_steps['feature_selection'].get_support()
                    selected_feature_names = [self.feature_names[i] for i, selected in enumerate(selected_features) if
                                              selected]
                else:
                    selected_feature_names = self.feature_names

                # Get coefficients
                coefs = abs(classifier.coef_[0])
                top_indices = np.argsort(coefs)[-10:][::-1]

                for i, idx in enumerate(top_indices):
                    feature_name = selected_feature_names[idx] if idx < len(
                        selected_feature_names) else f"feature_{idx}"
                    coef_value = classifier.coef_[0][idx]
                    print(f"   {i + 1:2d}. {feature_name:<15} : {coef_value:8.4f}")

        except Exception as e:
            print(f" Feature importance not available: {e}")

    def predict_sample(self, sample_data):
        #Predict a new sample using the trained pipeline.
        if self.best_model is None:
            print("Model not trained. Please run the analysis first.")
            return None

        # Convert to DataFrame with proper column names
        if isinstance(sample_data, list):
            sample_df = pd.DataFrame([sample_data], columns=self.feature_names)
        else:
            sample_df = pd.DataFrame(sample_data, columns=self.feature_names)

        # Make predictions using the full pipeline
        prediction = self.best_model.predict(sample_df)[0]
        probabilities = self.best_model.predict_proba(sample_df)[0]

        # Convert prediction back to class name
        diagnosis = self.class_names[prediction]
        confidence = max(probabilities)

        print(f"\nPREDICTION RESULTS:")
        print(f"   Diagnosis: {diagnosis}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Probabilities:")
        for i, class_name in enumerate(self.class_names):
            print(f"     {class_name}: {probabilities[i]:.4f}")

        return {
            'diagnosis': diagnosis,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities
        }

    def run_complete_analysis(self, filepath=None, use_feature_selection=True):
        """Run complete analysis using scikit-learn best practices."""
        print("Run complete analysis")
        print("Using Scikit-learn Best Practices")

        # 1. Load and process data
        self.load_data(FILEPATH)

        # 2. Create ML pipeline
        self.create_pipeline(use_feature_selection=use_feature_selection)

        # 3. Split data
        self.split_data()

        # 4. Optimize hyperparameters
        grid_search = self.optimize_hyperparameters()

        # 5. Evaluate model
        results = self.evaluate_model()

        print(f"\nANALYSIS COMPLETE!")
        print(f"Best Model: Logistic Regression")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"ROC-AUC Score: {results['roc_auc']:.4f}")
        print(f"Model can be used for predictions!")



        return results, grid_search




# Example usage
if __name__ == "__main__":
    print("SCIKIT-LEARN BREAST CANCER CLASSIFICATION")

    # Create classifier
    classifier = BreastCancerClassifier(random_state=42)

    # Run complete analysis
    # For your data file: results, grid_search = classifier.run_complete_analysis('your_file.data')
    results, grid_search = classifier.run_complete_analysis()

    # Example prediction
    print(f"\n EXAMPLE PREDICTION")
    print("=" * 30)
    sample_features = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                       1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                       25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
    sample_features2 = [11.81,17.39,75.27,428.9,0.1007,0.05562,0.02353,0.01553,0.1718,0.0578,0.1859,1.926,1.011,14.47,0.007831,0.008776,0.01556,0.00624,0.03139,0.001988,12.57,26.48,79.57,489.5,0.1356,0.1,0.08803,0.04306,0.32,0.06576]
    prediction = classifier.predict_sample(sample_features2)