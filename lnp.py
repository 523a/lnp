# -*- coding: utf-8 -*-
import numpy
import itertools
import torch
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
roberta.eval()
from fairseq.data.data_utils import collate_tokens
fillchar="_"



lnp = Flask(__name__)

@lnp.route("/")
def home():
    return render_template("index.html")

@lnp.route("/get")
def get_bot_response():
    
    userText = request.args.get('msg')
    #формироывние таблицы    
    userText = userText.replace('\n',' ').replace('\t','').replace('\xa0','')#.replace(',','')
    tokenText=sent_tokenize(userText, language="russian")
    sizetoken = len(tokenText)
    batch_of_pairs= list(itertools.combinations(tokenText,2))
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    roberta.to(device)
    
    batch = collate_tokens([roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs[0:100]], pad_idx=1)
    
    logprobs = roberta.predict('mnli', batch)
    resh = logprobs.argmax(dim=1).detach().numpy()


    #вывод таблицы
    outp=[]
    for i in range(int((sizetoken*(sizetoken-1))/2)):
          outp.append( '|'+batch_of_pairs[i][0].ljust(50,fillchar)[:50]+'|'+batch_of_pairs[i][1].ljust(50,fillchar)[:50]+'|'+str(resh[i])+'|'+'\n')
          
    out = ''.join(outp)
    return (out[1:])



if __name__ == "__main__":
    lnp.run(host = '0.0.0.0', port=5100)#(host = '0.0.0.0', port=5100)
