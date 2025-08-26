# EPL PROPHET 🏆⚽

**Predictive and Explanatory Analytics System for English Premier League**

[![Model Accuracy](https://img.shields.io/badge/Accuracy-53.7%25-green)](analysis1.0/outputs/champion_results.csv)
[![ML Framework](https://img.shields.io/badge/ML-Random_Forest-blue)](analysis1.0/models/final_breakthrough.py)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-orange)](analysis1.0/models/explainer_fixed.py)
[![Web App](https://img.shields.io/badge/Web-Flask-red)](app.py)

## 🎯 Project Overview

EPL Prophet is an advanced machine learning system that predicts English Premier League match outcomes with **53.7% accuracy** while providing complete explanations for each prediction. The system combines traditional football analytics with modern ML techniques and financial market indicators.

## 🏗️ Architecture

### Phase 1: Data Foundation
- **Data Standardization** (`data_standardization.py`)
- **Elo Rating System** (`ANALYSIS1.0/models/elo_rating_system.py`)
- **Expected Goals (xG) Analysis** (`xg_analysis_system.py`)
- **Advanced Feature Engineering** (`advanced_feature_engineering.py`)

### Phase 2: Team-as-Stocks System
- **Recency Weighting** (`recency_weighted_system.py`)
- **Exponential Moving Averages (EMA)**
- **Momentum & MACD-style Indicators**
- **Multi-timeframe Ensemble** (`phase2_multi_timeframe_ensemble.py`)

### Phase 3: Breakthrough Optimization
- **Hyperparameter Optimization** (`final_breakthrough.py`)
- **Logarithmic Ratio Features** (Key Innovation!)
- **Champion Random Forest Model** (53.7% accuracy)
- **Feature Selection & Ensemble**

### Phase 4: Explainability & Deployment
- **SHAP Explanations** (`explainer_fixed.py`)
- **Dynamic Fixture Fetching** (`fixture_fetcher.py`)
- **Flask Web Application** (`app.py`)
- **Real-time Predictions**

## 🧠 Key Innovations

### 1. **Logarithmic Ratios** (Breakthrough Feature)
```python
log_ratio = np.log((home_metric + 1) / (away_metric + 1))
```
- Captures extreme advantages between teams
- Most predictive feature type in our model
- Handles zero values gracefully

### 2. **Team-as-Stocks Approach**
- EMA-based form tracking
- Momentum indicators (3, 5, 10 match windows)
- Volatility measures
- Technical analysis crossovers

### 3. **Multi-timeframe Ensemble**
- Short-term (5 matches): 52.5% accuracy
- Medium-term (10 matches): 51.8% accuracy  
- Long-term (20 matches): 50.9% accuracy
- Dynamic weighting based on form

## 📊 Model Performance

| Model | Accuracy | Key Features |
|-------|----------|-------------|
| **Champion RF** | **53.7%** | Logarithmic ratios, EMA features |
| XGBoost | 52.1% | Tree-based ensemble |
| Gradient Boost | 51.4% | Boosting approach |
| Neural Network | 49.8% | Deep learning |

### Top Predictive Features:
1. `log_elo_advantage` (17.2% importance)
2. `log_form_advantage` (14.8% importance)
3. `log_xg_advantage` (12.3% importance)
4. `momentum_crossover` (9.1% importance)
5. `ema_volatility_ratio` (7.6% importance)

## 🔍 Explainability (SHAP)

Every prediction comes with detailed explanations:
- **Feature Attribution**: Which factors drove the prediction
- **Direction Impact**: Positive/negative influence
- **Magnitude**: How much each factor matters
- **Confidence Intervals**: Prediction uncertainty

Example:
```
Manchester City vs Liverpool
Prediction: Man City Win (68% confidence)

Key Factors:
+ Elo Advantage: +0.23 (City rated higher)
+ Home Advantage: +0.15 (Playing at Etihad)
+ Form Advantage: +0.11 (City's recent form)
- xG Efficiency: -0.08 (Liverpool creating better chances)
```

## 🌐 Web Application

Interactive Flask web app for match predictions:
- **Match Selection**: Choose from upcoming fixtures
- **Live Predictions**: Real-time probability calculations
- **Detailed Explanations**: SHAP-powered insights
- **Visual Dashboard**: Charts and statistics
- **Mobile Responsive**: Works on all devices

### Running the Web App:
```bash
cd epl-prophet-web
python app.py
# Visit http://localhost:8080
```

## 📁 Project Structure

```
EPL_PROPHET/
├── data/                          # Season CSV files (14/15 - 25/26)
├── ANALYSIS1.0/
│   ├── models/                    # All ML models and analysis
│   │   ├── elo_rating_system.py
│   │   ├── xg_analysis_system.py
│   │   ├── final_breakthrough.py  # Champion model
│   │   └── explainer_fixed.py     # SHAP explainability
│   └── outputs/                   # Generated datasets and models
│       ├── champion_rf.joblib     # 53.7% accurate model
│       ├── champion_features.csv  # Final dataset
│       └── champion_results.csv   # Performance metrics
├── epl-prophet-web/              # Deployment package
│   ├── app.py                    # Flask web application
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS/JS assets
│   └── requirements.txt          # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/RCRRTOW3R2/EPL_PROPHET.git
cd EPL_PROPHET
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
# or for web app:
cd epl-prophet-web
pip install -r requirements.txt
```

### 3. Run Analysis (Optional)
```bash
python data_standardization.py
cd ANALYSIS1.0/models
python final_breakthrough.py
```

### 4. Start Web App
```bash
cd epl-prophet-web
python app.py
```

## 📈 Development Timeline

- **Week 1**: Data standardization & Elo system
- **Week 2**: xG analysis & feature engineering  
- **Week 3**: Team-as-stocks & recency weighting
- **Week 4**: Multi-timeframe ensemble
- **Week 5**: Hyperparameter optimization & logarithmic breakthrough
- **Week 6**: SHAP explainability
- **Week 7**: Web deployment & API integration

## 🔮 Future Enhancements

### Technical Improvements:
- **Real-time APIs**: Live odds, lineups, injuries
- **Player-level Analysis**: Individual performance metrics
- **Weather Integration**: Match conditions impact
- **Social Sentiment**: Fan confidence tracking

### Model Enhancements:
- **Deep Learning**: LSTM for sequence modeling
- **Ensemble Stacking**: Meta-learning optimization
- **Bayesian Approaches**: Uncertainty quantification
- **Transfer Learning**: Cross-league adaptability

### Application Features:
- **Mobile App**: Native iOS/Android
- **Live Updates**: In-match prediction updates
- **Portfolio Tracking**: Multiple league support
- **API Service**: Third-party integrations

## 🎯 Business Applications

- **Sports Betting**: Informed wagering decisions
- **Fantasy Sports**: Player selection optimization
- **Media Content**: Data-driven match previews
- **Team Analytics**: Performance improvement insights
- **Fan Engagement**: Enhanced viewing experience

## 📊 Data Sources

- **Historical Match Data**: 11 seasons (2014-2026)
- **Team Statistics**: Goals, shots, possession, cards
- **Market Odds**: Betting market implied probabilities
- **Expected Goals**: Shot-based xG calculations
- **Referee Data**: Official assignments and bias

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Football-Data.org**: Historical match data
- **SHAP Library**: Model explainability framework
- **Scikit-learn**: Machine learning foundation
- **Flask**: Web application framework
- **Bootstrap**: UI/UX components

---

**EPL Prophet** - *Making Football Predictions Transparent and Actionable* ⚽🔮

*Built with ❤️ for the beautiful game* 