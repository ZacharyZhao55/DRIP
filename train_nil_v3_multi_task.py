"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""
import math
from torch.utils.data import DataLoader
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import torch
import random

# filename = ['2003 - pnnl','2006 - stewards','2007 - nlm','2007 - puget sound','2007 - water use','2009 - library']
#
# for name in filename:
#     print(name)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
print(model_name)
train_batch_size =64          #128 The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 100
num_epochs = 30

# Save path of the model
model_save_path = 'output/training_nli_multi-task_v3_(3)_(batch_8)'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Here we define our SentenceTransfor mer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
print(word_embedding_model.get_word_embedding_dimension())
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),  pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)
# pooling_model1 = models.Pooling(word_embedding_model.get_word_embedding_dimension(),  pooling_mode_mean_tokens=True,
#                                pooling_mode_cls_token=True,
#                                pooling_mode_max_tokens=False)
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# model1 = SentenceTransformer(modules=[word_embedding_model, pooling_model1])
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Check if CUDA is available ans switch to GPU
print(torch.version.cuda)
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

#Check if dataset exsist. If not, download and extract  it
# nli_dataset_path = r'C:\Users\zhaoziyan\Documents\开题文件\不完整需求\需求条目化\data\siamese-bert\data/bertTrain.tsv.gz'
nli_dataset_path = r'Data/train/bertTrain.tsv.gz'
sts_dataset_path = r'Data/train/bertTest.tsv.gz'

# nli_dataset_path = r'C:\Users\zhaoziyan\Documents\开题文件\不完整需求\需求条目化\data\siamese-bert\data/gemini-train.tsv.gz'
# sts_dataset_path = r'C:\Users\zhaoziyan\Documents\开题文件\不完整需求\需求条目化\data\siamese-bert\data/stsbenchmark.tsv.gz'

if not os.path.exists(nli_dataset_path):
    util.http_get('https://sbert.net/datasets/AllNLI.tsv.gz', nli_dataset_path)

if not os.path.exists(sts_dataset_path):
    print('dev数据不存在')
    # util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

label12int = {"0": 0, "1": 1}
train_samples = []
traincos_samples = []
train1_samples = []
with gzip.open(nli_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'train':
            # print(row['label-cos'])
            score = float(row['label-cos']) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

            train_samples.append(inp_example)

            labelcos_id = label12int[row['label-conj']]
            traincos_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=labelcos_id))

            label1_id = label12int[row['label1']]
            train1_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=label1_id))


train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss_sts = losses.CosineSimilarityLoss(model=model)


# Special data loader that avoid duplicates within a batch
# train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)
traincos_dataloader = DataLoader(traincos_samples, shuffle=True, batch_size=train_batch_size)
traincos_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label12int))



# Our training loss
# train_loss = losses.MultipleNegativesRankingLoss(model)
train1_dataloader = DataLoader(train1_samples, shuffle=True, batch_size=train_batch_size)
train1_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label12int))





#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

print(len(train_dataloader))
# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss_sts),(train1_dataloader,train1_loss),(traincos_dataloader,traincos_loss)],
          # evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'test':
            score = float(row['score']) / 5.0 #Normalize score to range 0 ... 1
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)
