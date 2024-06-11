import csv
import os, sys
import pandas as pd
import scipy as sp
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import movie_reviews
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, classification_report
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, dataset
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from SmilesPE.tokenizer import *
from smiles_pair_encoders_functions import *
from itertools import chain, repeat, islice
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
import json
from sst_reg import *
from tqdm import tqdm
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

'''
Initialize tokenizer
'''
vocab_file = 'VocabFiles/vocab_spe.txt'
spe_file = 'VocabFiles/SPE_ChEMBL.txt'
tokenizer = SMILES_SPE_Tokenizer(vocab_file=vocab_file, spe_file= spe_file)

def pad_infinite(iterable, padding=None):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
   return islice(pad_infinite(iterable, padding), size)

def tokenize_function(examples, tokenizer, ntoken) :
    return np.array(list(pad(tokenizer(examples)['input_ids'], ntoken, 0)))

def dataloader_inference(data, smilescol, batch, ntoken, tokenizer):
    tqdm.pandas()
    smiles_df = pd.DataFrame(data = {'text': data[smilescol]})
    smiles_df['text'] = smiles_df['text'].apply(lambda x: tokenize_function(x, tokenizer=tokenizer, ntoken=ntoken))
    features_tensor = torch.tensor(np.stack([tok_dat for tok_dat in smiles_df['text']]))
    dataloader = DataLoader(features_tensor, batch_size=batch, shuffle=False, num_workers=1, pin_memory=True)
    return dataloader

#############################################################
parser = ArgumentParser()#add_help=False)

parser.add_argument(
    "-m", "--modelwghts", type=Path, required=True, help="Path for model weights"
)

parser.add_argument(
    "-L", "--location", type=Path, required=True, help="Location for compound data"
)

parser.add_argument('-D', '--datasets',
                    type=lambda s: re.split(' |, ', s),
                    required=True,
                    help='comma or space delimited list of datasets to screen'
                    )

#parser.add_argument(
#    "-D", "--datasets", type=json.loads, required=True, help="Datasets to screen"
#)

parser.add_argument(
    "-s", "--smilescol", type=str, required=True, help="Column for SMILES"
)

parser.add_argument(
    "-c", "--confmod", type=Path, required=True, help="config file for model"
)

parser.add_argument(
    "-b", "--batch", type=int, required=True, help="batch size"
)

args = parser.parse_args()
with open(args.confmod, 'r') as f:
        config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wghts = torch.load(args.modelwghts)
print(wghts)
if True:
    model = TransformerModel(ntoken=config['ntoken'], d_model=config['d_model'], nhead=config['nhead'], d_hid=config['d_hid'],
                     nlayers=config['nlayers'], dropout= config['dropout'], device=device)
    
    model.load_state_dict(wghts['model_state_dict'])

model = model.to(device)
# Ensure the model is in evaluation mode
model.eval()

All_Files = np.array([])
All_Dirs = np.array([])
for dirs in args.datasets:
    list_dir_files = np.array(sorted(os.listdir(f'{args.location}/{dirs}')))
    All_Files = np.concatenate((All_Files, list_dir_files))
    dir_enumerate = np.array([dirs for i in range(len(list_dir_files))]) 
    All_Dirs = np.concatenate((All_Dirs, dir_enumerate))

split_files = np.array_split(All_Files, size)[rank]
split_dirs = np.array_split(All_Dirs, size)[rank]

cutoff=9
for fil, dirs in tqdm(zip(split_files, split_dirs)):
    data = pd.read_csv(f'{args.location}/{dirs}/{fil}')
    dataloader = dataloader_inference(data, args.smilescol, args.batch, config['ntoken'], tokenizer)
    preds_fil = []
    with torch.no_grad():
        for j, batch_X in enumerate(dataloader):
            preds = model(batch_X.to(device)).flatten()
            preds_fil.extend(preds.cpu().detach().numpy())
            del(preds)

    SMILES_DS = np.vstack((data[args.smilescol], np.array(preds_fil).flatten())).T
    SMILES_DS = sorted(SMILES_DS, key=lambda x: x[1], reverse=True)
    filtered_data = list(OrderedDict((item[0], item) for item in SMILES_DS if item[1] >= cutoff).values())
    filename = f'output/{dirs}/{os.path.splitext(fil)[0]}.dat'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['smiles', 'score'])
        writer.writerows(filtered_data)

