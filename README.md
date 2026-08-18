Predictive Maintenance Machine Learning Pipeline

Project Summary

Developed an end-to-end machine learning system and real-time telemetry simulator to predict, diagnose, and prevent industrial equipment failures, minimizing unplanned operational downtime.
Multi-Model Development: Built and compared baseline Logistic Regression, Random Forest, and XGBoost gradient boosting classifiers to predict multi-class mechanical failure modes (Heat Dissipation, Overstrain, and Tool Wear) with high predictive accuracy.

Data Preprocessing & Scaling: Implemented rigorous data preprocessing workflows utilizing scikit-learn for feature scaling (StandardScaler), train-test splitting, and stratifying imbalanced operational sensor data.

Real-time IoT Telemetry Simulator: Engineered a live monitoring stream utilizing probabilistic inference (predict_proba) to evaluate incoming sensor metrics minute-by-minute and trigger automated safety thresholds ahead of imminent equipment failure.

Feature Importance Analysis: Performed post-hoc model interpretability and feature importance analysis via XGBoost to identify primary mechanical wear drivers, such as torque limits and operational temperatures.
