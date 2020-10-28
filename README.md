# Alzheimer's-Disease-Prediction-using-NLP
This is a research project. The goal of the project is design a model which can predict whether a person is suffereing from Alzheimer's r not based on his/her way of speaking.
To achieve this goal, I have designed CNN-LSTM with attention model and also implemented CNN, and CNN-LSTM without attention for this task. I have used the models using pretrained
 embedding (Glove) and randomly initialised embedding.

# Dataset
The data used for this project is Dementia Bank Dataset.

# Models used

# CNN

This is a simple CNN model used for text classification.

# CNN-LSTM

This is a hybrid model. The input is passed through 1D convolution and then it is passed through Bidirectional LSTM.

# CNN-LSTM with Attention

This model is also a hybrid model. The attention layer is added which helps the model to focus on specific parts of the input sentence which helps the model to classify accurately.

# WORD2VEC PLOT

I also trained the WORD2VEC model on the words given in the dataset. This plot verifies that all the words are mostly related in the embedding space 
as almost all the transcripts are the descriptions of few scenes so all the words are related.

# EVALUATION METRICS

Metrics used for the model evaluation are :-
1. Accuracy
2. Precision
3. Recall
4. Specificity
5. F1 score
6. AUC
|MODELS|TECHNIQUE|ACCURACY %|PRECISION|RECALL|F1 SCORE|SPECIFICITY|AUC|
|------|--------|-----|---------|------|--------|-----------|---|
|CNN|Randomly initialized embedding|86.74|0.8671|0.8501|0.8585|0.8830|0.9105|

# RESULTS

CNN-LSTM with attention achieved the highest accuracy of 87.51% on the test data.
