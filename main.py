import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib as jb

df = pd.read_csv(r"C:\Users\LENOVO\Documents\AI\MachineLearning\Digital_Literacy\data\digital_literacy_dataset.csv")
df.drop(columns=["User_ID"], axis=1, inplace=True)
df['Education_Level'].fillna('Unknown', inplace=True)

le = LabelEncoder()
categorical_columns = ["Gender", "Education_Level", "Employment_Status", "Household_Income", "Location_Type", "Engagement_Level", "Employment_Impact"]
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])


x = df.drop(columns=["Overall_Literacy_Score"])
print(x.columns)
y = df["Overall_Literacy_Score"]
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_Train, Y_Train)
pred_value = model.predict(X_Test)

mae = mean_absolute_error(Y_Test, pred_value)
mse = mean_squared_error(Y_Test, pred_value)
rmse = np.sqrt(mse)  
r_squared = r2_score(Y_Test, pred_value)
print(f"Mean_Absolute_Error: {mae}")
print(f"Root_Mean_Squared_Error: {rmse}")
print(f"R2_Score: {r_squared}")

input_values = [25, 1, 2, 1, 2, 1, 8, 7, 9, 10, 9, 10, 5, 2.5, 8, 4, 2, 7, 9, 8, 1]
new_pred = model.predict([input_values])
print(f"Overall_Literacy_Score: {round(new_pred[0])}")

jb.dump(model, r"C:\Users\LENOVO\Documents\AI\MachineLearning\Digital_Literacy\Model.pkl")
jb.dump(le, r'C:\Users\LENOVO\Documents\AI\MachineLearning\Digital_Literacy\LabelEncoder.pkl')

