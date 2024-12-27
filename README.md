# Project breif
Developed an end-to-end machine learning pipeline for density and viscosity prediction in synthetic deep eutectic solvents, achieving state-of-the-art R² scores of 99.23% (density) and 98.45% (viscosity).

Engineered a robust CI/CD pipeline using GitHub Actions to automate model training and deployment, ensuring seamless integration and continuous delivery on AWS EC2.

Fine-tuned 10 advanced ML models, leveraging hyperparameter optimization to identify the most performant solution for predictive accuracy.


# Density and Viscosity Prediction in Synthetic Deep Eutectic Solvents
This project focuses on building an end-to-end machine learning pipeline to predict the density and viscosity of synthetic deep eutectic solvents with unprecedented accuracy. Leveraging advanced regression techniques, rigorous feature engineering, and robust model evaluation, the project achieved state-of-the-art R² scores of 99.23% for density and 98.45% for viscosity.

## Key Highlights

Model Fine-Tuning: Experimented with and fine-tuned 10 advanced regression models, including XGBoost, Random Forest, and Ridge Regression, to identify the optimal predictors.

CI/CD Integration: Implemented a Continuous Integration and Continuous Deployment (CI/CD) pipeline using GitHub Actions to automate training, testing, and deployment processes.

AWS Deployment: Successfully deployed the final model on an AWS EC2 instance for real-time inference and scalability.

Feature Engineering: Applied advanced preprocessing techniques, such as outlier removal, scaling, and feature selection, to enhance model performance.
      
## Repository Structure

/notebooks: Jupyter notebooks for exploratory data analysis and model development.

/src: Core scripts for preprocessing, model training, and evaluation.

/models: Saved models and pickle files for reproducibility.

/deployment: Scripts and configurations for deploying the model on AWS EC2.

CI/CD Pipeline: YAML workflows for GitHub Actions integration.

## Results

Achieved R² of 99.23% for density prediction and 98.45% for viscosity, setting a benchmark in predictive modeling for deep eutectic solvents.

Feel free to explore the repository and its components to understand the methodology and implementation. For any questions or collaboration opportunities, please reach out via LinkedIn or create an issue in the repository.
