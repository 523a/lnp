# -*- coding: utf-8 -*-
#  начальный код взят отсюда
# https://github.com/pytorch/fairseq/tree/master/examples/roberta


import numpy
import itertools
import torch
#from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request
import dash
import dash_html_components as html
#import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()

from fairseq.data.data_utils import collate_tokens
fillchar="_"

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

lnp = Flask(__name__)

@lnp.route("/")
def home():
    return render_template("index.html")

@lnp.route("/get")
def get_bot_response():
      
      userText = request.args.get('msg')
    
      #анализ текста и разбиение его на токены

      userText=userText.strip("\t").strip("\t").strip("\t")
      tokens = userText.strip( ).split('\n')
      for i in range(len(tokens)):
          #   тупо режем токены
          tokens[i]=tokens[i][:200]


      sizetoken = len(tokens)

      #формироывние таблицы    
      #userText = userText.replace('\n',' ').replace('\t','').replace('\xa0','')#.replace(',','')
      batch_of_pairs= list(itertools.combinations(tokens,2))
      print(batch_of_pairs)
      
      device = "cuda" if torch.cuda.is_available() else "cpu"
      roberta.to(device)
      
      batch = collate_tokens([roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs[0:100]], pad_idx=1)
      
      logprobs = roberta.predict('mnli', batch)
      resh = logprobs.argmin(dim=1).detach().numpy()
        
        
      #вывод таблицы
      outp=[]
      outp1=numpy.array
      for i in range(int((sizetoken*(sizetoken-1))/2)):
            outp.append( '|'+batch_of_pairs[i][0].ljust(50,fillchar)[:50]+'|'+batch_of_pairs[i][1].ljust(50,fillchar)[:50]+'|'+str(resh[i])+'|'+'\n')
            outp1( [batch_of_pairs[i][0].ljust(50,fillchar)[:50],batch_of_pairs[i][1].ljust(50,fillchar)[:50],str(resh[i])])
      
        
      out = ''.join(outp)
      
      
        
      return (out[1:-2])
        


if __name__ == "__main__":
    lnp.run(host = '0.0.0.0', port=5200)#(host = '0.0.0.0', port=5100)
