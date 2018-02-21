import pickle

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

NAME = 'main-0_3.35'
IM_SIZE = 90

print('Load and test model "{}"'.format(NAME))

# load test data
x_test = []
csv_test = pd.read_csv('../input/sample_submission.csv')
for f in tqdm(csv_test['id'].values):
    img = cv2.imread('../input/test/{}.jpg'.format(f))
    x_test.append(cv2.resize(img, (IM_SIZE, IM_SIZE)))
x_test_raw  = np.array(x_test, np.float32) / 255.

# load the model
model_name = 'bestmodel_{}.h5'.format(NAME)
print('Try to load model file "{}"'.format(model_name))
model = load_model('../output/{}'.format(model_name))
model.summary()

# predict on tests data
preds = model.predict(x_test_raw, verbose=1)

# save prediction to csv
out_file = 'predicted-{}.csv'.format(NAME)
classes = csv_test.columns.values[1:]
frame = pd.DataFrame(preds, index=csv_test['id'].tolist(), columns=classes)
frame.to_csv("../output/{}".format(out_file), index_label='id') 
print('Export prediction in "{}"'.format(out_file))