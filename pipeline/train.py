import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import joblib



# Завантаження даних
df = pd.read_csv("C:/Users/Asus/Desktop/Навчання 3 курс/2 семестр/Інформаційні технології смарт систем/Cellphone.csv")

# Підготовка даних
X = df.drop(["Product_id", "Sale", "Price"], axis=1)
y = df["Price"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Тренування моделі
dtree = DecisionTreeRegressor(random_state=42)
start = time.process_time()
dtree.fit(X_train, y_train)
end = time.process_time()

# Оцінка моделі
train_score = dtree.score(X_train, y_train)
test_score = dtree.score(X_test, y_test)
train_rmse = mean_squared_error(y_train, dtree.predict(X_train), squared=False)
test_rmse = mean_squared_error(y_test, dtree.predict(X_test), squared=False)
train_mae = mean_absolute_error(y_train, dtree.predict(X_train))
test_mae = mean_absolute_error(y_test, dtree.predict(X_test))

# Виведення результатів
print("DecisionTreeRegressor trained ✓ in", round(end - start, 3), "sec")
print("Train Score:", train_score)
print("Test Score:", test_score)
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train MAE:", train_mae)
print("Test MAE:", test_mae)

# Збереження моделі
joblib.dump(dtree, 'dtree_model.sav')