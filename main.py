import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = 'digital_marketing_campaign_dataset (1).csv'  # Update with your actual file path
df = pd.read_csv(file_path)

# Preprocessing the data
# Dropping the CustomerID column as it is not useful for prediction
df = df.drop(columns=['CustomerID'])

# Separating features and target
X = df.drop(columns=['Conversion'])
y = df['Conversion']

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
joblib.dump(label_encoders['Gender'], 'label_encoder_Gender.joblib')
joblib.dump(label_encoders['CampaignChannel'], 'label_encoder_CampaignChannel.joblib')
joblib.dump(label_encoders['CampaignType'], 'label_encoder_CampaignType.joblib')
joblib.dump(label_encoders['AdvertisingPlatform'], 'label_encoder_AdvertisingPlatform.joblib')
joblib.dump(label_encoders['AdvertisingTool'], 'label_encoder_AdvertisingTool.joblib')


# Training the Gradient Boosting model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Saving the model using joblib
model_filename = 'gradient_boosting_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")
