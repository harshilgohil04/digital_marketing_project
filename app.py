import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('gradient_boosting_model.joblib')

# Load the label encoders used during training
label_encoders = {
    'Gender': joblib.load('label_encoder_Gender.joblib'),
    'CampaignChannel': joblib.load('label_encoder_CampaignChannel.joblib'),
    'CampaignType': joblib.load('label_encoder_CampaignType.joblib'),
    'AdvertisingPlatform': joblib.load('label_encoder_AdvertisingPlatform.joblib'),
    'AdvertisingTool': joblib.load('label_encoder_AdvertisingTool.joblib')
}

# Function to preprocess the input data
def preprocess_input(data):
    for column, le in label_encoders.items():
        data[column] = le.transform([data[column]])[0]
    return data

# Streamlit app
st.title('Marketing Campaign Conversion Prediction')

# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=30)
gender = st.selectbox('Gender', options=['Male', 'Female'])
income = st.number_input('Income', min_value=0)
campaign_channel = st.selectbox('Campaign Channel', options=['Social Media', 'Email', 'PPC'])
campaign_type = st.selectbox('Campaign Type', options=['Awareness', 'Retention', 'Conversion'])
ad_spend = st.number_input('Ad Spend', min_value=0.0)
click_through_rate = st.number_input('Click Through Rate', min_value=0.0)
conversion_rate = st.number_input('Conversion Rate', min_value=0.0)
website_visits = st.number_input('Website Visits', min_value=0)
pages_per_visit = st.number_input('Pages Per Visit', min_value=0.0)
time_on_site = st.number_input('Time On Site', min_value=0.0)
social_shares = st.number_input('Social Shares', min_value=0)
email_opens = st.number_input('Email Opens', min_value=0)
email_clicks = st.number_input('Email Clicks', min_value=0)
previous_purchases = st.number_input('Previous Purchases', min_value=0)
loyalty_points = st.number_input('Loyalty Points', min_value=0)
advertising_platform = st.selectbox('Advertising Platform', options=['IsConfid'])
advertising_tool = st.selectbox('Advertising Tool', options=['ToolConfid'])

# Create a dictionary from the input fields
input_data = {
    'Age': age,
    'Gender': gender,
    'Income': income,
    'CampaignChannel': campaign_channel,
    'CampaignType': campaign_type,
    'AdSpend': ad_spend,
    'ClickThroughRate': click_through_rate,
    'ConversionRate': conversion_rate,
    'WebsiteVisits': website_visits,
    'PagesPerVisit': pages_per_visit,
    'TimeOnSite': time_on_site,
    'SocialShares': social_shares,
    'EmailOpens': email_opens,
    'EmailClicks': email_clicks,
    'PreviousPurchases': previous_purchases,
    'LoyaltyPoints': loyalty_points,
    'AdvertisingPlatform': advertising_platform,
    'AdvertisingTool': advertising_tool
}

# Preprocess the input data
input_data = preprocess_input(input_data)
input_df = pd.DataFrame([input_data])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    result = 'Converted' if prediction == 1 else 'Not Converted'
    st.write(f"The model predicts: **{result}**")

