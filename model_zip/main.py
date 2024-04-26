from model import attention_lstm_random

if __name__ == "__main__":
    # initialize the model
    att = attention_lstm_random()
    att_model = att.att_model

    # inference,predict
    test_txt = " well  the little boy's in the cookie jar . and he's about to fall on the floor because the chair's tilted . mom's drying dishes . she's not paying attention because the water's running over sink's running over . water's all over the floor . the little girl I think is begging her brother to give her a cookie . I'm not sure about that . must be summer time because the window's open . you can see the the grass and shrubbery outside . and see a few dishes that mom has already dried . "
    Y_pred = att_model.predict(att.preprocess_text(test_txt))
    probability = Y_pred[0][0]

    print("Probability of the text being a dementia is: ", probability)
