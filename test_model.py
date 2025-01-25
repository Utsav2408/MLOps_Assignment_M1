import joblib
import numpy as np

def test_model_loading():
    # Load the model
    model = joblib.load("model.joblib")
    assert model is not None, "Model not loaded properly"

def test_model_prediction():
    # Load the model
    model = joblib.load("model.joblib")
    
    # Test sample input
    test_input = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(test_input)
    
    assert len(prediction) == 1, "Prediction output shape is incorrect"
    print("Model prediction test passed")

if _name_ == "_main_":
    test_model_loading()
    test_model_prediction()