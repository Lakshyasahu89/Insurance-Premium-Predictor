import pickle
from pathlib import Path

import pandas as pd



# import the ml model
model_path = Path(__file__).with_name('model.pkl')
with model_path.open('rb') as f:
    model = pickle.load(f)
    
# MLFlow se aata hai ye 
MODEL_VERSION = '1.0.0'


# Get class labels from model ( important for matching probabilities to class names)
class_labels = model.classes_.tolist()

def predict_output(user_input: dict):
    
    df = pd.DataFrame([user_input])
    
    #predict the class
    predicted_class = model.predict(df)[0]
    
    # get probabilities for all classes
    probabilities = model.predict_proba(df)[0]
    confidence = max(probabilities)
    
    # Create mapping : {class_name: probability}
    class_probs = dict(zip(class_labels,map(lambda p: round(p,4), probabilities)))
    
    return{
        'predicted_class': predicted_class,
        'confidence': round(confidence, 4),
        'probabilities': class_probs
    }
    
