import os 
import sys 
sys.path.append('src') 
from predict import load_model, predict_session 
def test_model_inference(): 
    model = load_model('models/logistic_regression_pipeline.joblib') 
    c, p, l = predict_session(model, {'score': 85, 'duration_min': 25, 'xp_earned': 50}) 
    assert c in [0, 1], 'Class must be binary 0 or 1' 
    print('Unit test passed: Model inference works as expected!') 
if __name__ == '__main__': 
    test_model_inference() 
