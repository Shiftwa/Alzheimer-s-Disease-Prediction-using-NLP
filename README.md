# Alzheimer's Disease Prediction using NLP
This is a research project. The goal of the project is design a model which can predict whether a person is suffereing from Alzheimer's or not based on his/her way of speaking.
To achieve this goal, I have designed CNN-LSTM with attention model and also implemented CNN, and CNN-LSTM without attention for this task. I have used the models using pretrained embedding (Glove) and randomly initialised embedding.

# Dataset
The data used for this project is Dementia Bank Dataset.

# Models used

# CNN

This is a simple CNN model used for text classification.

# CNN-LSTM

This is a hybrid model. The input is passed through 1D convolution and then it is passed through Bidirectional LSTM.

# CNN-LSTM with Attention

This model is also a hybrid model. The attention layer is added which helps the model to focus on specific parts of the input sentence which helps the model to classify accurately.


# EVALUATION METRICS

Metrics used for the model evaluation are :-
1. Accuracy
2. Precision
3. Recall
4. Specificity
5. F1 score
6. AUC

|MODELS|Embedding used|ACCURACY %|PRECISION|RECALL|F1 SCORE|SPECIFICITY|AUC|
|------|------------------------------|-----|------|------|------|------|------|
|CNN   |Randomly initialized embedding|86.74|0.8671|0.8501|0.8585|0.8830|0.9105|
|CNN   |Glove embedding|87.05|0.8655|0.8599|0.8627|0.8801|0.9136|
|CNN - LSTM  |Randomly initialized embedding|87.05|0.8679|0.8566|0.8622|0.8830|0.9128|
|CNN - LSTM   |Glove embedding|87.21|0.8589|0.8729|0.8659|0.8713|0.9122|
|CNN - LSTM with Attention  |Randomly initialized embedding|85.51|0.8585|0.8306|0.8443|0.8771|0.9176|
|CNN - LSTM with Attention  |Glove embedding|**87.51**|**0.8717**|0.8631|**0.8674**|**0.8859**|**0.9222**|

# RESULTS

CNN-LSTM with Attention using Glove Embedding achieved the highest accuracy of 87.51% on the test data.
