# Full ML Pipeline — OOP & Modular Architecture

Two problems:
- **Classification**: Online Course Engagement → predict `CourseCompletion` (binary)
- **Regression**: Student Performance Factors → predict `Exam_Score` (continuous)

## Proposed Architecture

```
assignment1/
├── utils/
│   ├── data_utils.py          (existing — data splitting)
│   ├── data_cleaner.py        (existing — DataCleaner class)
│   ├── eda.py                 [NEW] EDAAnalyzer class
│   ├── feature_engineer.py    [NEW] FeatureEngineer class
│   ├── model_trainer.py       [NEW] ModelTrainer class
│   └── model_evaluator.py     [NEW] ModelEvaluator class
├── classification/
│   ├── pipeline.py            [NEW] ClassificationPipeline
│   └── app.py                 (existing — Streamlit deployment)
├── regression/
│   └── pipeline.py            [NEW] RegressionPipeline
└── main.py                    [MODIFY] Orchestrator entry point
```

---

## Proposed Changes

### Dependencies

#### Install `scikit-learn` and `matplotlib`/`seaborn`
```bash
pip install scikit-learn matplotlib seaborn
```

---

### Utils Layer

#### [NEW] [eda.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/utils/eda.py)
`EDAAnalyzer` class:
- `summary()` — shape, dtypes, describe, nulls
- `plot_distributions()` — histograms for numeric cols
- `plot_correlations()` — heatmap
- `plot_target_distribution()` — bar/hist of target
- `plot_categorical_counts()` — bar charts for categorical cols
- Saves all plots to `outputs/{problem}/eda/`

#### [NEW] [feature_engineer.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/utils/feature_engineer.py)
`FeatureEngineer` class:
- `encode_categoricals()` — LabelEncoder / OneHotEncoder
- `scale_numerics()` — StandardScaler / MinMaxScaler
- `create_features()` — problem-specific derived features
- `transform()` — full pipeline

#### [NEW] [model_trainer.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/utils/model_trainer.py)
`ModelTrainer` class:
- [train(X, y, model)](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/utils/data_utils.py#16-42) — fit a model
- `train_multiple(X, y, models_dict)` — train & compare multiple algorithms
- `tune_hyperparameters(X, y, model, param_grid)` — GridSearchCV
- `save_model(path)` / `load_model(path)` — joblib serialization

#### [NEW] [model_evaluator.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/utils/model_evaluator.py)
`ModelEvaluator` class:
- For classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix plot, ROC curve plot
- For regression: MSE, RMSE, MAE, R², residual plot, actual-vs-predicted plot
- `compare_models()` — side-by-side metrics table
- Saves all plots to `outputs/{problem}/evaluation/`

#### [MODIFY] [data_cleaner.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/utils/data_cleaner.py)
Already created earlier — no changes needed.

---

### Pipeline Layer

#### [NEW] [classification/pipeline.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/classification/pipeline.py)
`ClassificationPipeline` — orchestrates the full classification workflow:
1. Load data → 2. EDA → 3. Clean → 4. Feature engineer → 5. Train (Logistic Regression, Random Forest, Gradient Boosting) → 6. Evaluate → 7. Tune best model → 8. Test → 9. Save model

Target: `CourseCompletion`, drop `UserID`

#### [NEW] [regression/pipeline.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/regression/pipeline.py)
`RegressionPipeline` — same flow but for regression:
Models: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor
Target: `Exam_Score`

#### [MODIFY] [main.py](file:///media/etbytes-lab/projects/AAU-AI/year-one/semester-two/ML/assignment1/main.py)
Entry point that runs both pipelines end-to-end.

---

## Verification Plan

### Automated Tests
- Run `python main.py` and verify no errors
- Check that `outputs/` contains EDA plots and evaluation plots
- Check that model `.pkl` files are saved
- Verify evaluation metrics print to console

### Manual Verification
- Inspect EDA plots visually
- Review classification/regression metrics
