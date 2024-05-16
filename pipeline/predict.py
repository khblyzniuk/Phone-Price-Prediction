import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Завантаження даних
new_data = pd.read_csv("C:/Users/Asus/Desktop/Навчання 3 курс/2 семестр/Інформаційні технології смарт систем/Cellphone.csv")

# Підготовка нових даних
X_new = new_data.drop(["Product_id", "Sale", "Price"], axis=1)
X_new = StandardScaler().fit_transform(X_new)

# Завантаження натренованої моделі
dtree_model = joblib.load("dtree_model.sav")  # Підставте шлях до файлу збереженої моделі

# Прогнозування на нових даних
predictions = dtree_model.predict(X_new)

# Виведення прогнозів
print("Predictions for new data:")
print(predictions)

# Створення DataFrame з прогнозами
predictions_df = pd.DataFrame(predictions, columns=["Predictions"])

# Збереження прогнозів у файл CSV
predictions_df.to_csv("predictions.csv", index=False)