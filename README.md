# MLOps: Iris Dataset End-to-End Project

This repository demonstrates a simple MLOps workflow using the **Iris dataset**. It includes everything for data exploration and model training to deployment and monitoring (simulated).

---

## Project Structure

```
.
├── data/                  # Raw and processed data
│   └── iris.csv
├── notebooks/             # EDA and modeling notebooks
│   └── eda.ipynb
├── scripts/               # Training and evaluation scripts
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── generate_data_card.py
├── models/                # Saved model files
│   └── iris_model.pkl
├── deployment/            # Simulated deployment files (e.g. FastAPI)
│   └── app.py
├── monitoring/            # Mock monitoring dashboard or metrics
│   └── monitoring_logs.csv
├── tests/                 # Unit and integration tests
│   └── test_train.py
├── iris_data_card.txt     # Auto-generated data card
├── iris_pairplot.png      # Visualization of dataset
├── Dockerfile             # Docker setup for deployment
├── requirements.txt       # Python dependencies
└── README.md              # You're here
```
