# ✈️ Airline Fare Prediction Using Machine Learning

A comprehensive machine learning project that predicts airline ticket prices using various flight parameters and historical data. This project implements multiple ML algorithms to provide accurate fare predictions for different airlines and routes.

## 📊 Project Overview

This project aims to predict airline ticket prices by analyzing various factors such as:
- Airlines and flight routes
- Departure and arrival times
- Flight duration
- Number of stops
- Seasonal patterns
- Booking timing

The model helps travelers make informed decisions about when to book flights and estimate travel costs.

## 🛠️ Technologies Used

- **Python 3.x**
- **Jupyter Notebook**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization
- **Web Scraping Tools** - Data collection

## 📁 Project Structure

```
Airline-Fare-Prediction/
├── FlightPricePrediction.ipynb          # Main prediction notebook
├── Airline fare prediction.ipynb        # Alternative prediction analysis
├── model training.ipynb                 # Model training and evaluation
├── Scraped_dataset.csv                 # Raw scraped data
├── Cleaned_dataset.csv                 # Cleaned and preprocessed data
├── processed_airline_dataset.csv       # Further processed dataset
├── preprocessed_dataset2.csv           # Additional preprocessing version
├── featureadded_dataset.csv           # Dataset with engineered features
└── README.md                           # Project documentation
```

## 📈 Dataset Information

The project uses multiple datasets representing different stages of data processing:

1. **Scraped_dataset.csv** - Raw data collected from airline websites
2. **Cleaned_dataset.csv** - Data after initial cleaning and validation
3. **processed_airline_dataset.csv** - Standardized and formatted data
4. **preprocessed_dataset2.csv** - Advanced preprocessing with outlier handling
5. **featureadded_dataset.csv** - Final dataset with engineered features

### Key Features:
- Airline names
- Source and destination cities
- Departure and arrival times
- Flight duration
- Number of stops
- Price (target variable)
- Date of journey
- Additional route information

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AbhishekG27/-Airline-Fare-Prediction-Using-Machine-Learning.git
cd Airline-Fare-Prediction-Using-Machine-Learning
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 📋 Usage

### 1. Data Preprocessing
Run the data cleaning and preprocessing steps:
```python
# Load and clean the raw dataset
python data_preprocessing.py
```

### 2. Feature Engineering
Execute feature engineering to create meaningful predictors:
- Time-based features (hour, day, month)
- Duration categories
- Route popularity metrics
- Price trend indicators

### 3. Model Training
Train multiple machine learning models:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting
- XGBoost
- Support Vector Regression

### 4. Model Evaluation
Evaluate models using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Cross-validation scores

### 5. Prediction
Use the trained model to predict flight prices:
```python
# Example prediction
predicted_price = model.predict([[airline, source, destination, duration, stops]])
```

## 🔬 Machine Learning Workflow

1. **Data Collection**: Web scraping from airline websites
2. **Data Cleaning**: Handle missing values, outliers, and inconsistencies
3. **Exploratory Data Analysis**: Visualize patterns and relationships
4. **Feature Engineering**: Create relevant features from raw data
5. **Model Selection**: Compare different algorithms
6. **Hyperparameter Tuning**: Optimize model performance
7. **Model Evaluation**: Assess accuracy and generalization
8. **Deployment**: Create prediction interface

## 📊 Key Insights

- **Seasonal Patterns**: Flight prices vary significantly by season
- **Booking Timing**: Advance booking generally offers lower prices
- **Route Popularity**: Popular routes have more price variation
- **Airline Differences**: Significant price differences between airlines
- **Day of Week Effect**: Weekend flights are typically more expensive

## 🎯 Model Performance

| Model | RMSE | R² Score | Accuracy (%) |
|-------|------|----------|--------------|
| **Random Forest** | 1593.50 | 0.8951 | 91.52 |
| **Extra Trees Regressor** | 1711.84 | 0.8789 | 91.50 |
| **Gradient Boosting** | 3816.96 | 0.9654 | - |
| **XGBoost** | 2806.74 | 0.6745 | 83.56 |
| **Linear Regression** | 3896.96 | 0.3726 | 73.01 |

### 🏆 Best Performing Models

1. **Random Forest** - Achieved the best overall performance with lowest RMSE (1593.50) and highest accuracy (91.52%)
2. **Extra Trees Regressor** - Close second with RMSE of 1711.84 and 91.50% accuracy
3. **Gradient Boosting** - Highest R² score (0.9654) but higher RMSE compared to tree-based models

**Key Findings:**
- Tree-based ensemble methods (Random Forest, Extra Trees) significantly outperform linear models
- Random Forest provides the most reliable predictions with consistent performance across metrics
- Linear Regression shows limitations in capturing non-linear price patterns
- The models achieve over 90% accuracy for the top performers, indicating strong predictive capability

## 📈 Future Improvements

- [ ] Real-time data integration
- [ ] Deep learning models (LSTM, Neural Networks)
- [ ] Weather data incorporation
- [ ] Fuel price correlation analysis
- [ ] Mobile app development
- [ ] API development for real-time predictions

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Abhishek G**
- GitHub: [@AbhishekG27](https://github.com/AbhishekG27)
- Project Link: [Airline Fare Prediction](https://github.com/AbhishekG27/-Airline-Fare-Prediction-Using-Machine-Learning.git)

## 🙏 Acknowledgments

- Airlines for providing data accessibility
- Open source community for ML libraries
- Contributors and supporters of this project

## 📞 Contact

For any questions or suggestions, please feel free to reach out through GitHub issues or email.

---

**⭐ Star this repository if you found it helpful!**
