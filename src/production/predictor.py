import joblib
import pandas as pd


# =========================
# LOAD MODEL
# =========================
model = joblib.load("../../models/production/model_rf.pkl")


# =========================
# ESTIMATE RESOURCES
# =========================
def estimate_resources(complexity):
    if complexity == "low":
        return 2, 1
    elif complexity == "medium":
        return 3, 2
    elif complexity == "high":
        return 5, 4
    else:
        return 3, 2  # default
    

# =========================
# PREDICT FUNCTION
# =========================
def predict_cost(data):
        
        required_fields = [
            'product_type', 'material_type',
            'length_m', 'width_m', 'height_m',
            'complexity_level', 'quantity',
            'finishing_type', 'installation',
            'location_type'
        ]

        for field in required_fields:
            if field not in data:
                raise ValueError(f"{field} is required")

            # =========================
            # HITUNG VOLUME
            # =========================
            volume = (data['length_m'] * data['width_m'] * data['height_m'])

            # =========================
            # AUTO GENERATE
            # =========================
            num_workers, production_days = estimate_resources(data['complexity_level'])

            # =========================
            # BENTUK DATAFRAME
            # =========================
            input_data = {
                'product_type': data['product_type'],
                'material_type': data['material_type'],
                'length_m': data['length_m'],
                'width_m': data['width_m'],
                'height_m': data['height_m'],
                'volume_m3': volume,
                'complexity_level': data['complexity_level'],
                'quantity': data['quantity'],
                'num_workers': num_workers,
                'production_days': production_days,
                'finishing_type': data['finishing_type'],
                'installation': data['installation'],
                'location_type': data['location_type']
            }

            df = pd.DataFrame([input_data])

            # =========================
            # PREDIKSI
            # =========================
            prediction = model.predict(df)[0]

            return{
                "predicted_cost": int(prediction),
                "estimated_workers": num_workers,
                "estimated_days": production_days
            }