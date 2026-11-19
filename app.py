from fastapi import FastAPI
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, MODEL_VERSION, model


app = FastAPI()

# human readable endpoints
@app.get('/')
def home():
    return{'message':'Insurance Premium Prediction API'}


#machine readable endpoints
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'model_version': MODEL_VERSION ,
        'model_loaded' : model is True
    }


@app.post('/predict' , response_model=PredictionResponse)
def predict_premium(data: UserInput):

    user_input = {
        'bmi': data.bmi,
        'age_group': data.age_group,
        'lifestyle_risk': data.lifestyle_risk,
        'city_tier': data.city_tier,
        'income_lpa': data.income_lpa,
        'occupation': data.occupation
    }

    try:
        
        prediction = predict_output(user_input)

        return {
            'predicted_category': prediction['predicted_class'],
            'confidence': prediction['confidence'],
            'class_probabilities': prediction['probabilities']
        }
    
    except Exception as e:
        return {'error': str(e)}




