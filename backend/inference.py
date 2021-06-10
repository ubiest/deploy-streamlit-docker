# backend/inference.py
import numpy as np
from pathlib import Path 

def predict(image):
    model_path = Path('../')
    classifier_model = "model_90_60_90.pkl"
    IMAGE_SHAPE = (224, 224,3)
    model_inference = load_learner(model_path/classifier_model)
    print(image)

    predictions, x_mask, percents = model_inference.predict(image)
    x_mask = x_mask.numpy()
    percents = percents.numpy()
    if predictions != [] :
        predicts = [str.title(x.replace('_', ' ')) for x in predictions]
        weights = [round(percents[element], 4) for element in (np.nonzero(x_mask))[0]]
        output = ' \n '.join([f'{pred} with a probability of {weight:.2%}.' for pred, weight in zip(predicts, weights)])
        output_dict = dict(zip(predicts, weights))
        return output, output_dict
    else :
        output = 'We did not find any room categories'
        output_dict= None
        return output, output_dict
