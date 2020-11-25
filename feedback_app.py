# Dash dependencies
import dash
import dash_core_components as dcc
import dash_html_components as html
# import plotly.express as px
from dash.dependencies import Input, Output, State

# Model dependencies
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle, os, math, re, sys, nltk, feather, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import tokenize

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import regularizers, optimizers
from keras.utils import CustomObjectScope, to_categorical
from keras.engine.topology import Layer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras import initializers as initializers, regularizers, constraints

# Customized libraries
sys.path.insert(0,'./data_model')
import AttentionLayer
from feedback import clean_str, wordToSeq, sentAttentionWeights, feedback_ranking, feedback_highlight 

# Instantiate the app and the server
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Custom functions to load model and data
def load_ingredients():
    # To load the trained model
    global model, word_index, nontext_test, text_test
    filepath = os.path.sep.join(['./data_model/HANMLP_200E64B_GRUwithReLU_l21e6_dropout5_rmsprop_lr1e5_811.h5'])
    model = load_model(filepath, custom_objects={'AttentionWithContext': AttentionLayer.Attention}, compile=False)
    global graph
    graph = tf.get_default_graph()

    with open('./data_model/word_index_data_test.pkl', 'rb') as f:
        data = pickle.load(f)
        word_index = data['word_index']
        nontext_test = data['nontext_test']
        text_test = data['text_test']
    return model, word_index, nontext_test, text_test

model, word_index, nontext_test, text_test = load_ingredients()

app.layout = html.Div([
    html.H2("Feedback"),
    html.Div(["Submit your essay: ",
        dcc.Textarea(id='my-input-text', value='', style={'width':'100%', 'height': 200})]),
    html.Div(["Submit your provided ID: ",
        dcc.Input(id='my-input-nontext', value='', type='number')]),
    html.Br(),
    html.Button('Get Feedback', id='submit-button', n_clicks=0),
    html.Br(),
    html.Br(),
    html.H5('Your essay with feedback: ', style={'color': 'dimgray'}),
    html.Div(id='my-output', style={'color': 'black', 'fontSize': 16, 'font-style': 'italic'}),
])

@app.callback(
    dash.dependencies.Output(component_id='my-output', component_property='children'),
    dash.dependencies.Input(component_id='submit-button', component_property='n_clicks'),
    dash.dependencies.State(component_id='my-input-text', component_property='value'),
    dash.dependencies.State(component_id='my-input-nontext', component_property='value'))
def update_output(n_clicks, my_input_text, my_input_nontext):
    # global html_output
    if n_clicks>0:
        if my_input_text is not None and my_input_text is not '':
            if my_input_nontext is not None and my_input_nontext is not '':
                my_input_nontext = int(my_input_nontext)
                if my_input_nontext <= nontext_test.shape[0]:
                    input_text = clean_str(my_input_text)
                    input_text_array = wordToSeq(input_text,word_index,max_senten_len=40,max_senten_num=30,max_features=200000)
                    input_text_array = np.resize(input_text_array, (1,30,40)) 
                    nontext_input_array = np.resize(nontext_test[my_input_nontext], (1, 40))
                    with graph.as_default():
                        pred = model.predict([nontext_input_array, input_text_array])[0][0]
                        sent, nopad_output_array = feedback_ranking(my_input_text, my_input_nontext)
                        html_output = feedback_highlight(sent, nopad_output_array)
                        if (pred >= 0 and pred <.25):
                            label = 'unlikely'
                        elif (pred >=.25 and pred <.5):
                            label = 'less likely'
                        elif (pred >=.5 and pred <.75):
                            label = 'likely'
                        else:
                            label = 'very likely'
                    # return 'There is {:.0f}% chance.'.format(round(pred*100)) + html_output
                    return 'This project is {} to be funded. There is {:.0f}% chance.'.format(label, round(pred*100)), html.Div([html.Iframe(srcDoc=html_output, width='100%', height=200)])
                else:
                    return 'ID is incorrect. Please submit your provided ID.'
            else:
                return 'Please submit your provided ID.'
        else:
            return 'Please submit your essay.'

if __name__ == '__main__':
    app.run_server(debug=True) #, port=8050, host='0.0.0.0'
    # app.run_server(dev_tools_hot_reload=False) #to turn off hot-reloading