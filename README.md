# 🚗 AutoValuate — AI Car Price Prediction & Market Analytics

A **Streamlit** web application that uses a machine learning model to predict used car prices and provide deep market analytics.

## ✨ Features

- **AI-Powered Price Prediction** — Estimate a car's market value based on brand, model, year, mileage, fuel type, and more
- **Market Analytics Dashboard** — Interactive charts and insights on the used car market
- **Scikit-learn Pipeline** — Trained ML pipeline with preprocessing and regression
- **Interactive UI** — Built with Streamlit and Plotly for a rich user experience

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Data Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Harsh-Patel-25/CarPricePrediction.git
cd CarPricePrediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

> **Note:** The trained model file (`car_price_model.pkl`) is not included in the repository due to its size.  
> You will need to train it using the provided dataset before running the app.

### 5. Run the app

```bash
streamlit run app.py
```

## 📂 Project Structure

```
CarPricePrediction/
├── app.py                  # Main Streamlit application
├── cleaned_car_data.csv    # Cleaned dataset used for training & analytics
├── requirements.txt        # Python dependencies
└── .gitignore
```

## 📊 Dataset

The dataset (`cleaned_car_data.csv`) contains used car listings with features like brand, model, year, fuel type, transmission, mileage, and selling price.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
