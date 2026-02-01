# Quick Start Guide 🚀

## Installation & Setup

### Step 1: Navigate to Project Directory
```bash
cd insurance-enrollment-prediction
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

This will install all required packages:
- pandas, numpy (data processing)
- scikit-learn (ML algorithms)
- xgboost, lightgbm (gradient boosting)
- matplotlib, seaborn (visualization)
- fastapi, uvicorn (API - optional)

**Note**: Installation may take 5-10 minutes depending on your internet connection.

---

## Running the Pipeline

### Option 1: Run Complete Pipeline (Recommended)
```bash
python main.py
```

This will:
1. ✅ Load and explore data
2. ✅ Generate visualizations
3. ✅ Train 5 different models
4. ✅ Evaluate and compare models
5. ✅ Save best model
6. ✅ Display results

**Expected Runtime**: 5-10 minutes

### Option 2: Run Individual Steps

**Data Processing Only:**
```bash
python main.py --mode preprocess
```

**Model Training Only:**
```bash
python main.py --mode train
```

**With Hyperparameter Tuning:**
```bash
python main.py --mode train --tune
```

**Model Evaluation Only:**
```bash
python main.py --mode evaluate
```

**Make Predictions:**
```bash
python main.py --mode predict
```

---

## Starting the API (Bonus Feature)

### Step 1: Ensure Model is Trained
First, run the pipeline to train and save the model:
```bash
python main.py
```

### Step 2: Start API Server
```bash
uvicorn api.main:app --reload
```

Or:
```bash
python api/main.py
```

### Step 3: Access API Documentation
Open your browser and visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Step 4: Test API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "marital_status": "Married",
    "salary": 75000.0,
    "employment_type": "Full-time",
    "region": "West",
    "has_dependents": "Yes",
    "tenure_years": 5.5
  }'
```

---

## Viewing Results

### Generated Files

After running the pipeline, check these directories:

**📊 Visualizations** (`notebooks/`):
- `enrollment_distribution.png`
- `numerical_distributions.png`
- `categorical_analysis.png`
- `correlation_heatmap.png`
- `confusion_matrix_*.png`
- `roc_curve_*.png`
- `feature_importance_*.png`

**🤖 Models** (`models/`):
- `xgboost.pkl` (best model)
- `xgboost_metrics.pkl` (performance metrics)

**📁 Processed Data** (`data/`):
- `X_train.csv`, `X_test.csv`
- `y_train.csv`, `y_test.csv`
- `scaler.pkl`, `label_encoders.pkl`

### Reading the Report

Open `report.md` to see:
- Data observations and insights
- Model performance comparison
- Feature importance analysis
- Business recommendations
- Future improvements

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "FileNotFoundError: employee_data.csv"
**Solution**: Ensure you're running commands from the project root directory:
```bash
cd insurance-enrollment-prediction
python main.py
```

### Issue: API won't start
**Solution**: Make sure the model is trained first:
```bash
python main.py  # Train model
uvicorn api.main:app --reload  # Then start API
```

### Issue: Installation takes too long
**Solution**: You can install packages individually if needed:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

---

## Next Steps

1. ✅ Review `report.md` for detailed analysis
2. ✅ Check visualizations in `notebooks/` directory
3. ✅ Test the API with different inputs
4. ✅ Explore the code in `src/` directory
5. ✅ Customize and extend the pipeline

---

## Need Help?

- 📖 Read the full [README.md](README.md)
- 📊 Check the [report.md](report.md) for analysis
- 💻 Review code comments in `src/` files
- 🌐 Visit API docs at http://localhost:8000/docs (when running)

---

**Happy Predicting! 🎯**
