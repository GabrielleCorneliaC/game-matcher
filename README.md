<<<<<<< HEAD
# 🎮 Game Rating Predictor (KNN)

An API-based application built with **Flask** that uses the **K-Nearest Neighbors (KNN)** algorithm to predict a video game's rating. The model is trained on a dataset containing player activity metrics (Plays, Playing, Backlogs, Wishlist) and game genres.

## ⚙️ Technologies Used
* **Language:** Python
* **Backend & API:** Flask, Flask-CORS
* **Machine Learning:** Scikit-Learn (KNeighborsRegressor, KFold, Train-Test Split)
* **Data Processing:** Pandas, NumPy, OpenPyXL
* **Frontend:** HTML, CSS, JavaScript

## ✨ Key Features
1. **Rating Prediction API:** The `/api/predict` endpoint accepts player metrics and genres, returning the KNN model's predicted rating.
2. **Nearest Neighbors Match:** Returns 5 games with the most similar characteristics based on Euclidean distance calculations.
3. **Automated Evaluation:** Includes a standalone script to evaluate the model using 5-Fold Cross-Validation (MAE & RMSE) and export the logic logs and results to Excel.

## 🚀 Installation & Usage

### 1. Setup Environment
```bash
# Clone this repository
git clone [https://github.com/your-username/game-rating-predictor.git](https://github.com/your-username/game-rating-predictor.git)
cd game-rating-predictor

# Install required dependencies
pip install flask flask-cors pandas numpy scikit-learn openpyxl
=======
# game-matcher
This project is a Game Rating Predictor, an API-based system built with Flask and a K-Nearest Neighbors (KNN) model to predict video game scores. It processes player interaction metrics (plays, playing, backlogs, wishlist) normalized via Min-Max Scaling and encodes genre categories.
>>>>>>> 4da1ada9b1b0d9b663ddaa0defdba4abf5559e89
