from django.shortcuts import render
from .forms import UserInputForm
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import os
from django.conf import settings

def load_model_and_scaler():
    model_path = os.path.join(settings.BASE_DIR, 'recommendations', 'models', 'model.h5')
    scaler_path = os.path.join(settings.BASE_DIR, 'recommendations', 'models', 'scaler.pkl')
    
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

def recommend_plans(model, user_input, scaler, df, top_n=3):
    # Scale the user input
    user_input_scaled = scaler.transform([user_input])
    
    # Predict the cluster
    predicted_cluster = np.argmax(model.predict(user_input_scaled), axis=-1)[0]
    
    # Filter the dataset to get users in the same cluster
    cluster_data = df[df['Cluster'] == predicted_cluster]
    
    # Calculate the usage similarity
    cluster_data['similarity'] = np.linalg.norm(cluster_data.drop(columns=['Cluster']).values - user_input, axis=1)
    
    # Get top N recommendations
    recommended_plans = cluster_data.sort_values(by='similarity').head(top_n)
    
    # Add a Total_Charge column
    recommended_plans['Total_Charge'] = (
        recommended_plans['Day_Charge'] +
        recommended_plans['Eve_Charge'] +
        recommended_plans['Night_Charge'] +
        recommended_plans['Intl_Charge']
    )
    
    # Exclude columns 'Cluster', 'similarity'
    columns_to_exclude = ['Cluster', 'similarity']
    columns_to_include = [col for col in recommended_plans.columns if col not in columns_to_exclude]
    recommended_plans = recommended_plans[columns_to_include]
    recommended_plans = recommended_plans.reset_index(drop=True)
    
    return recommended_plans

def user_input_view(request):
    if request.method == 'POST':
        form = UserInputForm(request.POST)
        if form.is_valid():
            user_input = [
                form.cleaned_data['Day_Mins'],
                form.cleaned_data['Day_Calls'],
                form.cleaned_data['Day_Charge'],
                form.cleaned_data['Eve_Mins'],
                form.cleaned_data['Eve_Calls'],
                form.cleaned_data['Eve_Charge'],
                form.cleaned_data['Night_Mins'],
                form.cleaned_data['Night_Calls'],
                form.cleaned_data['Night_Charge'],
                form.cleaned_data['Intl_Mins'],
                form.cleaned_data['Intl_Calls'],
                form.cleaned_data['Intl_Charge'],
            ]
            
            # Calculate the total charge for the input plan
            input_total_charge = (
                form.cleaned_data['Day_Charge'] +
                form.cleaned_data['Eve_Charge'] +
                form.cleaned_data['Night_Charge'] +
                form.cleaned_data['Intl_Charge']
            )
            
            # Load model and scaler
            model, scaler = load_model_and_scaler()
            
            # Load dataset
            df = pd.read_csv(os.path.join(settings.BASE_DIR, 'recommendations', 'models', 'Reprocessed_Clustered_data.csv'))
            
            # Get recommendations
            recommended_plans = recommend_plans(model, user_input, scaler, df)
            
            # Render the results template with the necessary context
            return render(request, 'recommendations/results.html', {
                'recommended_plans': recommended_plans,
                'input_plan': user_input,
                'input_total_charge': input_total_charge,
            })
    else:
        form = UserInputForm()

    return render(request, 'recommendations/user_input.html', {'form': form})
