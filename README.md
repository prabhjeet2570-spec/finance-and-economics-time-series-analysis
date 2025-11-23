# 📈 Finance & Economics Time-Series Analysis

A comprehensive machine learning project for predicting S&P 500 stock prices using economic indicators and technical analysis.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## 🎯 Project Overview

This project implements multiple machine learning models to forecast S&P 500 closing prices using historical market data and macroeconomic indicators from 2000-2008. The goal is to build accurate predictive models that can help inform investment decisions.

**Key Achievement:** Our best model (Gradient Boosting) achieved **1.27% MAPE**, representing **97.4% improvement** over baseline predictions.

## 📊 Dataset

- **Source:** [Kaggle - Finance & Economics Dataset (2000-present)](https://www.kaggle.com/datasets/khushikyad001/finance-and-economics-dataset-2000-present)
- **Size:** 3,000 daily records
- **Time Period:** 2000-2008
- **Features:** 24 variables including:
  - Stock indices (S&P 500, NASDAQ, Dow Jones)
  - Price data (Open, Close, High, Low, Volume)
  - Economic indicators (GDP, Inflation, Unemployment, Interest Rates)
  - Consumer metrics (Confidence Index, Retail Sales)
  - Commodity prices (Crude Oil, Gold)

## 🏗️ Project Structure

```
finance-timeseries-analysis/
│
├── Finance_Economics_Analysis.ipynb    # Complete analysis notebook (all code)
└── README.md                            # Project documentation
```

**Note:** All code, analysis, and visualizations are contained in a single Colab notebook for easy execution and reproducibility.

## 🔧 Installation

### Prerequisites
- Google Colab account (recommended) OR
- Python 3.8+ with Jupyter Notebook

### Option 1: Google Colab (Recommended - No Setup Required!)

1. Download the notebook:
```bash
git clone https://github.com/yourusername/finance-timeseries-analysis.git
```

2. Upload `Finance_Economics_Analysis.ipynb` to [Google Colab](https://colab.research.google.com/)

3. Run all cells! The notebook will automatically:
   - Install required packages
   - Download the dataset from Kaggle
   - Generate all visualizations
   - Display results

### Option 2: Run Locally

1. Clone the repository:
```bash
git clone https://github.com/yourusername/finance-timeseries-analysis.git
cd finance-timeseries-analysis
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow kagglehub
```

3. Open and run the notebook:
```bash
jupyter notebook Finance_Economics_Analysis.ipynb
```

## 🚀 Usage

### Quick Start (Google Colab)
1. Open the notebook in [Google Colab](https://colab.research.google.com/)
2. Click **Runtime** → **Run all**
3. Wait 5-10 minutes for complete execution
4. All visualizations will be displayed inline
5. Results and metrics will be printed at the end

### What the Notebook Does
The notebook is organized into **8 sections**:

1. **Setup & Data Loading** - Import libraries and download dataset
2. **Data Cleaning** - Handle missing values and preprocess data
3. **Exploratory Data Analysis** - Generate correlation heatmaps and trend plots
4. **Feature Engineering** - Create 28 technical and economic features
5. **Time Series Decomposition** - Analyze trend, seasonality, and stationarity
6. **Model Development** - Train 5 models (Baseline, Ridge, RF, GB, LSTM)
7. **Model Evaluation** - Compare performance metrics
8. **Visualization** - Generate 9 comprehensive charts

### Expected Runtime
- **Google Colab (Free Tier)**: ~5-10 minutes
- **Local Machine (CPU)**: ~10-15 minutes
- **Local Machine (GPU)**: ~5-8 minutes

## 📐 Methodology

### 1. Data Preprocessing
- Handle missing values using forward-fill for time series continuity
- Sort data chronologically
- Focus on S&P 500 index for cleaner analysis

### 2. Feature Engineering (28 Features Created)
**Critical**: Uses **lagged features** to prevent data leakage
- **Previous Day Prices**: `Prev_Close`, `Prev_Open`, `Prev_High`, `Prev_Low`
- **Technical Indicators**: Moving Averages (5, 10, 20, 50-day), MA Ratios
- **Volatility Measures**: Rolling standard deviations (5, 20-day)
- **Returns**: 1-day, 5-day, 20-day price changes
- **Volume Indicators**: Volume ratios and moving averages
- **Lag Features**: Multiple time-step lags (1, 2, 3, 5, 10 days)
- **Time Features**: Day of week, month, quarter

### 3. Train-Test Split
- **Training**: 80% (789 samples, 2000-2006)
- **Testing**: 20% (198 samples, 2006-2008)
- **Method**: Chronological split (no random shuffling)

### 4. Models Implemented

| Model | Type | Purpose |
|-------|------|---------|
| Persistence Baseline | Simple | Benchmark (predicts yesterday's price) |
| Ridge Regression | Linear | Fast, interpretable baseline |
| Random Forest | Ensemble | Handles non-linear relationships |
| **Gradient Boosting** | **Ensemble** | **Best performer** |
| LSTM Neural Network | Deep Learning | Captures sequential patterns |

## 📊 Results

### Model Performance Comparison

| Model | MAE ($) | RMSE ($) | MAPE (%) | vs Baseline |
|-------|---------|----------|----------|-------------|
| Baseline (Persistence) | 1347.26 | 1654.99 | 56.24% | - |
| Ridge Regression | 81.91 | 105.97 | 3.69% | ↑ 93.9% |
| Random Forest | 54.23 | 72.89 | 1.93% | ↑ 96.0% |
| **Gradient Boosting** | **35.46** | **48.55** | **1.27%** | **↑ 97.4%** ⭐ |
| LSTM | 2109.97 | 2413.28 | 64.20% | ↓ -56.6% |

### 🏆 Best Model: Gradient Boosting
- **MAE**: $35.46 (average prediction error)
- **MAPE**: 1.27% (percentage error)
- **Interpretation**: For S&P 500 at ~$3000, predictions are within ±$38 (1.27%)
- **Performance Grade**: ⭐ **Excellent** (MAPE < 2% is outstanding for financial forecasting)

### Top 5 Most Important Features
1. `MA_10_ratio` (0.9052) - 10-day moving average ratio
2. `MA_10` (0.0802) - 10-day moving average
3. `MA_5` (0.0060) - 5-day moving average
4. `MA_5_ratio` (0.0020) - 5-day moving average ratio
5. `Price_Range_Ratio` (0.0008) - Daily price range ratio

## 📈 Key Insights

### Business Insights
✅ **Moving average ratios** are the strongest predictors of price movements  
✅ **Technical indicators** outperform macroeconomic factors for short-term forecasting  
✅ **Ensemble methods** (RF, GB) are more reliable than deep learning for this task  
✅ **Price momentum** and trend-following strategies show promise  
✅ **Volatility patterns** help identify risk periods  

### Technical Learnings
✅ **Data leakage prevention** is critical - only use historical data  
✅ **Feature engineering** matters more than model complexity  
✅ **Time-series splits** must preserve chronological order  
✅ **Baseline models** establish essential performance benchmarks  
✅ **Standardization** improves convergence for linear models  

## 🎨 Visualizations

The notebook automatically generates 9 comprehensive visualizations:

| Visualization | Description |
|---------------|-------------|
| 📈 Stock Price Trends | Historical price movements for S&P 500, NASDAQ, Dow Jones |
| 🔥 Correlation Heatmap | Feature relationships and multicollinearity analysis |
| 📊 Economic Indicators | GDP, inflation, unemployment rate trends over time |
| 📉 Time Series Decomposition | Trend, seasonal, and residual components |
| 🧠 LSTM Training History | Loss and MAE convergence plots |
| 🎯 Model Predictions Comparison | Actual vs predicted prices for each model |
| 📊 Model Metrics Comparison | Bar charts comparing MAE, RMSE, MAPE |
| 📉 Prediction Error Distribution | Histogram of prediction errors by model |
| ⭐ Feature Importance | Top 15 features ranked by importance score |

All visualizations are displayed inline in the notebook and automatically saved as high-resolution PNG files.

## ⚠️ Limitations

- Models trained on 2000-2008 data (pre-financial crisis period)
- Cannot predict black swan events or market crashes
- Past performance doesn't guarantee future results
- Transaction costs and slippage not modeled
- Real-time deployment requires live data feeds
- LSTM underperformed (needs hyperparameter tuning)

## 🚀 Future Improvements

1. **Sentiment Analysis** - Integrate news/social media sentiment
2. **Additional Technical Indicators** - MACD, Bollinger Bands, RSI improvements
3. **Ensemble Stacking** - Combine predictions from multiple models
4. **Walk-Forward Validation** - Rolling window cross-validation
5. **Multi-Period Testing** - Validate on bull/bear markets and crisis periods
6. **Transaction Cost Modeling** - Realistic backtesting with fees
7. **LSTM Optimization** - Grid search for optimal architecture
8. **Real-Time Pipeline** - Deploy model with live data API

## 📁 Repository Contents

- **Finance_Economics_Analysis.ipynb** - Complete Jupyter notebook with all code and analysis
- **README.md** - This documentation file

**Dataset:** Automatically downloaded from Kaggle when running the notebook (no manual download needed)

## 📚 References & Resources

### Dataset
- [Finance & Economics Dataset (2000-present)](https://www.kaggle.com/datasets/khushikyad001/finance-and-economics-dataset-2000-present) - Kaggle

### Libraries Used
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models
- **Statsmodels** - Time series analysis
- **TensorFlow/Keras** - Deep learning (LSTM)
- **KaggleHub** - Dataset download

### Learning Resources
- "Python for Finance" by Yves Hilpisch
- Scikit-learn Time Series Documentation
- TensorFlow LSTM Tutorial
- Kaggle Time Series Forecasting Courses

## 🙏 Acknowledgments

- Kaggle for providing the dataset
- Data Science training camp instructors for guidance
- Open-source community for ML libraries (scikit-learn, TensorFlow)
