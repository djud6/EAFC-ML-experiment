# 🎮⚽ EA FC Player Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Accuracy](https://img.shields.io/badge/R²-97.9%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Predicting EA Sports FC player prices with 97.9% accuracy using machine learning**

[📊 Research Paper](#) • [🚀 Quick Start](#quick-start) • [📈 Results](#results) • [💻 Demo](#demo)

</div>

---

## 📋 Overview

This project uses **XGBoost** to predict player prices in EA Sports FC 25 (formerly FIFA Ultimate Team) based on:
- ⚡ 45 performance statistics (pace, shooting, dribbling, etc.)
- 🎯 27 playstyles and 26 tactical roles
- 📅 Temporal market patterns
- 🏆 Player metadata (position, club, league, nation)

**Trained on 477,699 player-price observations** with exceptional performance:
- **97.9% R²** on test set
- **13.71% MAPE** (Mean Absolute Percentage Error)
- **24,088 coins MAE** (Mean Absolute Error)

---

## ✨ Key Features

- 🎯 **Highly Accurate**: Explains 97.9% of price variance
- ⚡ **Fast Inference**: <100ms per prediction
- 🔄 **Production Ready**: Saved model with pickle encoders
- 📊 **Interpretable**: Feature importance analysis reveals price drivers
- 🌐 **Deployable**: Easy to integrate into web apps or trading bots
- 📈 **Temporal Aware**: Captures seasonal market patterns

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/eafc-price-predictor.git
cd eafc-price-predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train Model

```bash
# Train on your data
python src/train.py --data data/player_price_history_85plus.csv
```

### Make Predictions

```python
from src.predict import EAFCPricePredictor

# Load model
predictor = EAFCPricePredictor()

# Predict single player
player_data = {
    'rating': 89,
    'position': 'ST',
    'stat_pace': 91,
    'stat_shooting': 90,
    # ... other attributes
}

price = predictor.predict_price(player_data)
print(f"Predicted price: {price:,} coins")
```

### Batch Predictions

```python
# Predict from CSV
predictor.predict_batch_from_csv(
    'new_players.csv', 
    'predictions.csv',
    prediction_date='2025-10-15'
)
```

---

## 📊 Results

### Model Performance

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **R²** | 0.984 | 0.978 | **0.979** |
| **MAE** | 20,884 | 23,981 | **24,088** |
| **RMSE** | 90,569 | 108,530 | **106,646** |
| **MAPE** | 12.39% | 13.71% | **13.71%** |

### Top Price Drivers

1. 🥅 **Rush Out Playstyle** (7.6%) - Goalkeeper premium
2. ⭐ **Overall Rating** (5.9%) - Core valuation factor
3. 👶 **Age** (3.4%) - Younger = more valuable
4. 🎯 **Target Forward Role** (3.3%) - Meta-dependent
5. 💪 **Bruiser Playstyle** (3.1%) - Physical presence
6. ⚡ **Sprint Speed** (2.0%) - Pace is king
7. 🎨 **Finesse Shot** (1.9%) - Gameplay effectiveness

### Sample Predictions

| Player | Rating | Position | Actual | Predicted | Error |
|--------|--------|----------|--------|-----------|-------|
| Lauren Hemp | 99 | LW | 148,222 | 142,009 | 4.2% |
| Lindsey Horan | 92 | CM | 82,507 | 97,140 | 17.7% |
| Laura Freigang | 86 | ST | 20,446 | 19,122 | 6.5% |

---

## 🏗️ Project Structure

```
eafc-price-predictor/
├── data/
│   ├── player_price_history_85plus.csv    # Training data
│   └── test_players.csv                   # Test data
├── models/
│   ├── eafc_price_predictor.json          # Trained model
│   ├── label_encoders.pkl                  # Categorical encoders
│   └── feature_columns.pkl                 # Feature schema
├── notebooks/
│   ├── 01_EDA.ipynb                       # Exploratory analysis
│   ├── 02_Training.ipynb                  # Model training
│   └── 03_Evaluation.ipynb                # Results & viz
├── src/
│   ├── train.py                           # Training script
│   ├── predict.py                         # Inference engine
│   └── utils.py                           # Helper functions
├── requirements.txt
├── README.md
└── research_paper.md                      # Full methodology
```

---

## 🔬 Methodology

### Data Pipeline

```
Raw Data (651k rows)
    ↓
Remove Zero Prices (478k rows)
    ↓
Feature Engineering (111 features)
    ↓
Train/Val/Test Split (70/15/15)
    ↓
XGBoost Training (1000 trees)
    ↓
Model Evaluation & Saving
```

### Key Techniques

- **Log Transformation**: `y = log(1 + price)` for stability
- **Label Encoding**: Handles 500+ clubs, 100+ nations
- **Temporal Features**: Month, week, day-of-week patterns
- **Early Stopping**: Prevents overfitting (50 rounds)
- **Regularization**: L1 (0.1) + L2 (1.0) for generalization

### Hyperparameters

```python
n_estimators: 1000
learning_rate: 0.05
max_depth: 8
subsample: 0.8
colsample_bytree: 0.8
```

---

## 📈 Visualizations

### Feature Importance
![Feature Importance](images/feature_importance.png)

### Predictions vs Actual
![Predictions](images/predictions_vs_actual.png)

### Residual Analysis
![Residuals](images/residual_analysis.png)

---

## 🎯 Use Cases

### For Players
- 🛒 **Smart Trading**: Identify undervalued players
- 💰 **Investment Timing**: Buy low, sell high
- 🏆 **Squad Building**: Find budget alternatives to expensive cards

### For Developers
- 🤖 **Trading Bots**: Automate arbitrage opportunities
- 📊 **Market Analysis**: Track price trends and patterns
- 🔮 **Price Forecasting**: Predict future market movements

### For Researchers
- 📚 **Virtual Economics**: Study digital market dynamics
- 🧪 **ML Applications**: Benchmark for tabular prediction
- 🎮 **Gaming Analytics**: Understand player valuation

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML utilities
- **Matplotlib/Seaborn** - Visualization

---

## 📚 Documentation

- [📄 Research Paper](research_paper.md) - Full methodology & results
- [🔧 API Documentation](docs/api.md) - Function reference
- [📖 User Guide](docs/user_guide.md) - Detailed usage examples
- [❓ FAQ](docs/faq.md) - Common questions

---

## 🚧 Future Improvements

- [ ] Ensemble with LightGBM and CatBoost
- [ ] Time series forecasting (LSTM/GRU)
- [ ] Confidence intervals (quantile regression)
- [ ] Real-time market data integration
- [ ] Web dashboard (Streamlit/Flask)
- [ ] Docker containerization
- [ ] REST API deployment

---

## 🤝 Contributing

Contributions welcome! Please check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Your Name**  
📧 Email: your.email@example.com  
💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)  
🌐 Portfolio: [yourwebsite.com](https://yourwebsite.com)

---

## 🙏 Acknowledgments

- EA Sports for creating EA FC
- FUTBIN & FUT.GG for price data
- XGBoost development team
- Scikit-learn contributors
- EA FC community

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ by [Your Name]**

[⬆ Back to Top](#-ea-fc-player-price-predictor)

</div>