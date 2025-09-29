import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("EA FC PLAYER PRICE PREDICTION MODEL")
print("="*60)

# 1. Load CSV
df = pd.read_csv("/content/sample_data/player_price_history_85plus.csv")
print(f"\n✓ Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 2. Drop rows where price == 0 or NaN
df = df[df["price"] > 0].copy()
print(f"✓ After removing zero prices: {df.shape[0]:,} rows")

# 3. Convert date → datetime, extract time features
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["month"] = df["date"].dt.month
df["week"] = df["date"].dt.isocalendar().week
df["day"] = df["date"].dt.day
df["dayofweek"] = df["date"].dt.dayofweek
df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)
print("✓ Time features extracted")

# 4. Extract height if it exists and is in string format
if "height" in df.columns and df["height"].dtype == "object":
    df["height_cm"] = df["height"].str.extract('(\d+)').astype(float)
    print("✓ Height converted to numeric")

# 5. Drop NAs
df = df.dropna(subset=["price"])

# 6. Encode categorical features and SAVE encoders
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
drop_cols = ["player_id", "player_name", "date", "height"]
categorical_cols = [col for col in categorical_cols if col not in drop_cols]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Save for later use

print(f"✓ Encoded {len(categorical_cols)} categorical features")

# 7. Define features
exclude_cols = ["player_id", "player_name", "date", "price", "height"]
features = [col for col in df.columns if col not in exclude_cols]

X = df[features].copy()
y = np.log1p(df["price"])  # log-transform

print(f"\n✓ Feature matrix: {X.shape[1]} features")
print(f"✓ Target variable: log-transformed price")

# 8. Train-validation-test split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
)

print(f"\n✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Validation set: {X_val.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# 9. Train XGBoost with early stopping
print(f"\n{'='*60}")
print("TRAINING MODEL...")
print("="*60)

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    tree_method="hist",
    early_stopping_rounds=50,
    eval_metric="rmse"
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=100
)

print(f"\n✓ Best iteration: {model.best_iteration}")

# 10. Predict on all sets
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Convert back from log space
y_train_actual = np.expm1(y_train)
y_val_actual = np.expm1(y_val)
y_test_actual = np.expm1(y_test)

y_train_pred_actual = np.expm1(y_train_pred)
y_val_pred_actual = np.expm1(y_val_pred)
y_test_pred_actual = np.expm1(y_test_pred)

# 11. Evaluate
print(f"\n{'='*60}")
print("MODEL PERFORMANCE")
print("="*60)

def print_metrics(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{set_name} Set:")
    print(f"  MAE:   {mae:>12,.0f} coins")
    print(f"  RMSE:  {rmse:>12,.0f} coins")
    print(f"  R²:    {r2:>12.4f}")
    print(f"  MAPE:  {mape:>12.2f}%")
    
    return {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}

train_metrics = print_metrics(y_train_actual, y_train_pred_actual, "Train")
val_metrics = print_metrics(y_val_actual, y_val_pred_actual, "Validation")
test_metrics = print_metrics(y_test_actual, y_test_pred_actual, "Test")

# Check for overfitting
print(f"\n{'='*60}")
if test_metrics["R²"] < train_metrics["R²"] - 0.1:
    print("⚠ Warning: Possible overfitting detected")
else:
    print("✓ Model generalizes well!")
print("="*60)

# 12. Feature Importance
print("\n" + "="*60)
print("TOP 20 FEATURE IMPORTANCES")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + feature_importance.head(20).to_string(index=False))

# Plot feature importance
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Top 20 features
feature_importance.head(20).plot(
    x='feature', y='importance', kind='barh', 
    ax=axes[0], legend=False, color='steelblue'
)
axes[0].set_xlabel('Importance (Gain)', fontsize=12)
axes[0].set_ylabel('')
axes[0].set_title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()

# Plot 2: Prediction vs Actual (Test Set)
sample_size = min(1000, len(y_test_actual))
sample_idx = np.random.choice(len(y_test_actual), sample_size, replace=False)
axes[1].scatter(y_test_actual.iloc[sample_idx], 
                y_test_pred_actual[sample_idx], 
                alpha=0.5, s=20, color='coral')
axes[1].plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 
             'k--', lw=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price', fontsize=12)
axes[1].set_ylabel('Predicted Price', fontsize=12)
axes[1].set_title(f'Predictions vs Actual (Test Set, n={sample_size})', 
                  fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Plots saved as 'model_analysis.png'")
plt.show()

# 13. Residual Analysis
residuals = y_test_actual - y_test_pred_actual
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Residual distribution
axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residual (Actual - Predicted)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Residuals vs Predicted
axes[1].scatter(y_test_pred_actual, residuals, alpha=0.5, s=20, color='mediumpurple')
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Price', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Residual plots saved as 'residual_analysis.png'")
plt.show()

# 14. Save model
model.save_model('eafc_price_predictor.json')
print(f"\n✓ Model saved as 'eafc_price_predictor.json'")

# 15. Example predictions
print(f"\n{'='*60}")
print("SAMPLE PREDICTIONS")
print("="*60)

sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
predictions_df = pd.DataFrame({
    'Player': df.loc[X_test.index[sample_indices], 'player_name'].values,
    'Actual_Price': y_test_actual.iloc[sample_indices].values,
    'Predicted_Price': y_test_pred_actual[sample_indices],
    'Error': np.abs(y_test_actual.iloc[sample_indices].values - y_test_pred_actual[sample_indices]),
    'Error_%': np.abs((y_test_actual.iloc[sample_indices].values - y_test_pred_actual[sample_indices]) / y_test_actual.iloc[sample_indices].values * 100)
})

print("\n" + predictions_df.to_string(index=False))

print(f"\n{'='*60}")
print("✓ MODEL TRAINING COMPLETE!")
print("="*60)