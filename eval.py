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

def tokenize_and_align_labels_for_pred(sentences,tokenizer,MAX_LENGTH):
  test_features={
      "input_ids":[],
      "attention_mask":[]
  }

  for sentence in sentences:
    sent_input_ids = []
    sent_attention_mask = []
    token_level_index= [] #index of the starting subword of a token

    for _, word in enumerate(sentence):
      tokenized = tokenizer(word, add_special_tokens=False)
      if len(sent_input_ids) ==0:
        token_level_index.append(0)
      else:
        token_level_index.append(len(sent_input_ids)) #save the index of the starting subword of a token 
       
      sent_input_ids.extend(tokenized["input_ids"])
      sent_attention_mask.extend(tokenized["attention_mask"])
   
    assert len(sentence) == len(token_level_index)

    #post-processing : matching the max length for the input_ids and attention_mask
    #input_ids : add cls,sep and padding tokens
    sent_input_ids = [2] + sent_input_ids+[3]
    for _ in range(MAX_LENGTH- len(sent_input_ids)):
      sent_input_ids.append(1) #add padding token
    #add attention mask for special tokens
    sent_attention_mask = [1]+ sent_attention_mask +[1] 
    for _ in range(MAX_LENGTH-len(sent_attention_mask)):
      sent_attention_mask.append(0)

    assert len(sent_input_ids) == MAX_LENGTH
    assert len(sent_input_ids) == len(sent_attention_mask)

    test_features["input_ids"].append(sent_input_ids)
    test_features["attention_mask"].append(sent_attention_mask)
    
  #convert to numpy arrays
  for key in test_features:
      test_features[key] = np.array([np.array(x) for x in test_features[key]]).astype('int32')

  return test_features,token_level_index


def predict(TEST_DATASET, FINETUNED,tokenizer,MAX_LENGTH):
  
  sentences, _ = split_text_label(TEST_DATASET)

  model = FINETUNED
  total_pred  = []

  for sentence in sentences:

    sent_pred = []

    dev_feature, token_level_index =tokenize_and_align_labels_for_pred([sentence],tokenizer,MAX_LENGTH)
    
    prediction = model.predict([dev_feature["input_ids"],dev_feature["attention_mask"]])["logits"][0]
    prediction = np.argmax(prediction, axis=1)

    #input = tokenizer.convert_ids_to_tokens(dev_feature["input_ids"][0])
    
    #write the prediction to the file 
    #format:  
    #sentence id
    #O
    #O
    #use the start subword index for writing the prediction
    for _,token_idx in enumerate(token_level_index):
      # note : consider that the CLS token is the first token at the input_id
      #if i!= len(token_level_index)-1:
        #print(input[token_idx+1: token_level_index[i+1]+1], " --> prediction : ", id_to_label[prediction[token_idx+1]])
      sent_pred.append(id_to_label[prediction[token_idx+1]])
    #print("-----------------")
    total_pred.append(sent_pred)

  assert len(sentences) == len(total_pred)
  assert len(sentences[0]) == len(total_pred[0])

  return total_pred

def get_sent_ids(filename):
  #sentence level dataset
  sent_ids = []
  with open(filename,'r',encoding="utf-8") as f:
    for line in f:
      if line.startswith("# id") is True: #a start of a new sentence
        sent_ids.append(line)

  return sent_ids


def write_prediction(DATA_DIR,FINETUNED,tokenizer,OUTPUT_DIR,MAX_LENGTH):
  TEST_DATASET = os.path.join(DATA_DIR,"ko_dev.conll")
  PRED_RESULT = os.path.join(OUTPUT_DIR,"ko_pred.conll")

  total_pred = predict(TEST_DATASET,FINETUNED,tokenizer,MAX_LENGTH)
  sent_ids = get_sent_ids(TEST_DATASET)

  with open(PRED_RESULT,'w', encoding='utf-8') as f:
    f.write(sent_ids[0])
    for i,pred in enumerate(total_pred):
      for token_pred in pred:
        f.write(token_pred+'\n')

      if i!= len(total_pred)-1:
        f.write("\n")
        f.write(sent_ids[i+1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The training data file. Expected to contain training/dev/test data")

    parser.add_argument("--pretrained_model", default=None, type=str, required=True,
                      help="Pretrained model for training from list: kobert, \n"
                      "krbert, koelectra, klue, xlm."
                      )
    parser.add_argument("--model_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="Where the model weights of what we want to evaluate is saved")
    parser.add_argument("--output_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="Where the evaluation result will be written")
    parser.add_argument("--max_seq_length",
                    default=256,
                    type=int,
                    help="The maximum total input sequence length for tokenization.")

    parser.add_argument("--create_chunks",
                default=1000,
                type=int,
                help="In the case when the eval dataset is large, do evaluation on chunks instead of sentence level")

   
    args = parser.parse_args()
    if args.pretrained_model == "koelectra":
        tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model =TFElectraForTokenClassification.from_pretrained('monologg/koelectra-base-v3-discriminator', num_labels=len(gs), from_pt=True)
        model.load_weights(args.model_dir)
        return model
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
        model.load_weights(args.model_dir)

    write_prediction(args.data_dir, model ,tokenizer,args.output_dir, args.max_seq_length)
if __name__ == '__main__':
    main()