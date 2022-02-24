import tensorflow as tf
import os
import numpy as np
from kobert_transformers import get_tokenizer
from transformers import ElectraTokenizer, AutoTokenizer,BertTokenizer
from transformers import TFElectraForTokenClassification,TFAutoModelForTokenClassification
from official import nlp 
import official.nlp.optimization
import argparse

gs = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
id_to_label={}
for key,value in gs.items():
  id_to_label[value]=key

def split_text_label(filename):
  sentences,labels = [],[]
  tokens,tags=[],[]
  f = open(filename,"r",encoding="utf-8")
  #read the lines and extract the data in ['word','entity'] format
  for line in f:
    if line.startswith("# id") is True: 
      #append the so far saved tokens
      if len(tokens)!=0 and len(tags)!=0:
        sentences.append(tokens)
        labels.append(tags)
      #initialize 
      tokens = [] 
      tags=[]
    elif len(line)>0 and line!="\n":
      line = line.strip()
      fields = line.split()
      tokens.append(fields[0])
      tags.append(fields[-1])
  #append the last sentence
  sentences.append(tokens)
  labels.append(tags)
  assert len(sentences) == len(labels)
  return sentences,labels

def tokenize_and_align_labels(FILENAME,tokenizer, MAX_LENGTH=256):
  """
  Given each sentence 's label and input tokens, turn it into a tokenized form 
  e.g.) tokens by bert tokenizer: ['_대한민국','만세'] --> ['_대한민국','_만','세']
        preserved labels : [1,2] -->[1,2,2]
  """
  sentences,labels = split_text_label(FILENAME)
  
  train_features={
      "input_ids":[],
      "attention_mask":[],
      "token_type_ids":[]
  }
  train_labels=[]

  for sentence, label in zip(sentences,labels):
    sent_input_ids = []
    sent_attention_mask = []
    sent_labels=[]

    for word_idx, word in enumerate(sentence):
      tokenized = tokenizer(word, add_special_tokens=False,)
      sent_input_ids.extend(tokenized["input_ids"])
      sent_attention_mask.extend(tokenized["attention_mask"])
      if len(tokenized["input_ids"])>1:
        #need to add the label to each of the word piece
        for _ in range(len(tokenized["input_ids"])):
          sent_labels.append(gs[label[word_idx]]) #propagate the labels
      else:
        sent_labels.append(gs[label[word_idx]]) #add the single label
       
    assert len(sent_input_ids) == len(sent_labels)

    #post-processing : matching the max length for the input_ids and attention_mask
    #input_ids : add cls,sep and padding tokens
    sent_input_ids = [2] + sent_input_ids+[3]
    for _ in range(MAX_LENGTH- len(sent_input_ids)):
      sent_input_ids.append(1) #add padding token
    #add labels for special token
    sent_labels = [12] + sent_labels+[12]
    for _ in range(MAX_LENGTH- len(sent_labels)):
      sent_labels.append(12) #add padding token labels
    #add attention mask for special tokens
    sent_attention_mask = [1]+ sent_attention_mask +[1]
    for _ in range(MAX_LENGTH-len(sent_attention_mask)):
      sent_attention_mask.append(0)

    assert len(sent_input_ids) == MAX_LENGTH
    assert len(sent_input_ids) == len(sent_labels)
    assert len(sent_input_ids) == len(sent_attention_mask)

    #debugging
    # for i in range(len(sent_input_ids)):
    #   if tokenizer.convert_ids_to_tokens(sent_input_ids[i])!="[PAD]":
    #     print(tokenizer.convert_ids_to_tokens(sent_input_ids[i]),sent_attention_mask[i], " ---> ", id_to_label[sent_labels[i]])
    # print("------------------")
    
    train_features["input_ids"].append(sent_input_ids)
    train_features["attention_mask"].append(sent_attention_mask)
    train_labels.append(sent_labels)
    
  assert len(train_features["input_ids"]) == len(train_labels)

  #convert to numpy arrays
  for key in train_features:
      train_features[key] = np.array([np.array(x) for x in train_features[key]]).astype('int32')
  train_labels = np.array([np.array(x) for x in train_labels]).astype('int32')

  return train_features, train_labels


def train_model(PRETRAINED,OUTPUT_DIR,tokenizer,DATA_DIR,learning_rate=5e-5,epochs=3,batch_size=16):
  TRAIN_DIR = os.path.join(DATA_DIR,"ko_train.conll")
  DEV_DIR = os.path.join(DATA_DIR,"ko_dev.conll")

  #get the bert-input format 
  train_features ={
      "input_ids":[],
      "attention_mask":[]
  }
  train_labels=[]
  #prepare bert training input
  train_features, train_labels =tokenize_and_align_labels(TRAIN_DIR,tokenizer,256)
  #dev set input
  dev_features, dev_labels = tokenize_and_align_labels(DEV_DIR,tokenizer,256)

  #prepare bert model
  model = PRETRAINED

  #adding learning rate warm up 
  train_data_size = len(train_features["input_ids"])
  steps_per_epoch = int(train_data_size / batch_size)
  num_train_steps = steps_per_epoch * epochs
  warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

  # creates an optimizer with learning rate schedule : adam weight decay - warms up from 0 and then decays to 0
  optimizer = nlp.optimization.create_optimizer(
      learning_rate, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
    
  model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
  )
  
  logdir = os.path.join(OUTPUT_DIR,"logs" )
  tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1, write_graph=True, write_grads=True, write_images=True )
  es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

  model.fit([train_features["input_ids"], train_features["attention_mask"]], train_labels, validation_data=(
        [dev_features["input_ids"], dev_features["attention_mask"]],
        dev_labels
    ),batch_size=batch_size, epochs=epochs,verbose=1, callbacks=[tensorboard_callback,es_callback])
  WEIGHTS = os.path.join(OUTPUT_DIR,"trained_model" )
  model.save_weights(WEIGHTS)
 


def main():
  parser = argparse.ArgumentParser()

  ## Required parameters
  parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The training data file. Expected to contain training/dev/test data")

  parser.add_argument("--pretrained_model", default=None, type=str, required=True,
                      help="Pretrained model for training from list: kobert, \n"
                      "krbert, koelectra, klue, xlm."
                      )
  parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="The maximum total input sequence length for tokenization.")

  parser.add_argument("--train_batch_size",
                      default=16,
                      type=int,
                      help="Batch size for training.")

  parser.add_argument("--learning_rate",
                      default=5e-5,
                      type=float,
                      help="The initial learning rate used in training.")
  parser.add_argument("--num_train_epochs",
                      default=3.0,
                      type=int,
                      help="Number of epochs used in training")

  parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="Output directory where model will be saved")
  args = parser.parse_args()

 

  if args.pretrained_model == "koelectra":
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model =TFElectraForTokenClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=len(gs), from_pt=True)
  else:
    if args.pretrained_model == "kobert":
      PRETRAINED = "monologg/kobert"
      from kobert_transformers import get_tokenizer
      tokenizer = get_tokenizer()
    elif args.pretrained_model == "krbert":
      tokenizer = BertTokenizer.from_pretrained('vocab_char_16424.txt', do_lower_case=False)
      PRETRAINED = "snunlp/KR-BERT-char16424"
    elif args.pretrained_model == "klue":
      tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large', do_lower_case=False)
      PRETRAINED = "klue/roberta-large"
    elif args.pretrained_model == "xlm":
      tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-large', do_lower_case=False)
      PRETRAINED = "xlm-roberta-large"
    else:
      print("Not supported model!")
    model = TFAutoModelForTokenClassification.from_pretrained(PRETRAINED, num_labels=len(gs),from_pt=True)

  #training 
  train_model(model,args.output_dir,tokenizer,args.data_dir, args.learning_rate, args.num_train_epochs, args.train_batch_size)

if __name__ == '__main__':
  main()