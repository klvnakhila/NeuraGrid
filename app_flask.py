from flask import Flask, render_template, jsonify
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
from appPred import predict_next_day, predict_next_7_days, get_weather_impact


# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/demand-forecast')
@app.route('/demand-forecast/<int:days>')
def demand_forecast(days=1):
    if days not in [1, 7]:
        error_msg = "Error: Invalid forecast duration. Only 1-day and 7-day forecasts are supported."
        return render_template('error.html', error=error_msg), 400
    
    try:
        if days == 1:
            gru_pred = np.array([predict_next_day()])
        elif days == 7:
            gru_pred = predict_next_7_days()
        
        # Generate timestamps for the predictions
        now = datetime.now()
        timestamps = [now + timedelta(days=i) for i in range(len(gru_pred))]
        
        # Prepare data for the template
        forecast_data = {
            'forecast_days': days,
            'timestamps': [ts.isoformat() for ts in timestamps],
            'gru_predictions': gru_pred.tolist(),
            'avg_demand': f"{np.mean(gru_pred):,.0f}",
            'change_pct': f"{((gru_pred[-1] - gru_pred[0]) / gru_pred[0] * 100):.1f}% change" if len(gru_pred) > 1 else "Single day forecast",
            'last_updated': datetime.now().strftime('%b %d, %Y %H:%M')
        }
        
        return render_template('prediction.html', forecast_data=forecast_data)
    
    except Exception as e:
        error_msg = f"Error generating forecast: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=error_msg), 500

@app.route('/api/forecast', methods=['GET'])
@app.route('/api/forecast/<int:days>', methods=['GET'])
def api_forecast(days=1):
    """API endpoint to get forecast data in JSON format"""
    if days not in [1, 7]:
        return jsonify({"error": "Invalid forecast duration. Only 1-day and 7-day forecasts are supported."}), 400
    
    try:
        if days == 1:
            gru_pred = np.array([predict_next_day()])
        elif days == 7:
            gru_pred = predict_next_7_days()
        
        now = datetime.now()
        timestamps = [(now + timedelta(days=i)).isoformat() for i in range(len(gru_pred))]
        
        return jsonify({
            "forecast_days": days,
            "timestamps": timestamps,
            "gru_predictions": gru_pred.tolist(),
            "avg_demand": float(np.mean(gru_pred)),
            "peak_demand": float(max(gru_pred)),
            "min_demand": float(min(gru_pred))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/weather-impact')
def weather_impact():
    try:
        weather = get_weather_impact()
        return render_template('weather.html', weather_data=weather)
    except Exception as e:
        error_msg = f"Error loading weather impact data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=error_msg), 500

@app.route('/weather-impact4')
def weather_impact4():
    try:
        weather = get_weather_impact()
        return render_template('weather4.html', weather_data=weather)
    except Exception as e:
        error_msg = f"Error loading weather impact data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return render_template('error.html', error=error_msg), 500
        
@app.route('/solar-forecast')
def solar_forecast():
    return render_template('components/solar.html')
if __name__ == '__main__':
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
