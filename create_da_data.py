"""
reads the train data and creates a new training data with (original sentences+augmented sentences)
on LOC, PER named entities
"""
import pandas as pd
import random
import argparse
import os


def split_text_label(DATASET):
    text={}
    sentences=[] 
    with open(DATASET,'r', encoding='utf-8') as f:
        key = None
        for line in f:
            if line.startswith("# id") is True:
                key = line.replace("\n","") #update the key
                text[key]=[]
                
            elif len(line)>0 and line!="\n":
                line =  line.strip()
                fields = line.split()
                text[key].append(fields[0])
    for key in text:
        sentence =" ".join(text[key])
        sentences.append(sentence)
    return text,sentences

def get_to_replace_NE(DATASET,ne_type):
    """ get the duplicated mentioned named entities from the training dataset"""
    if ne_type =="loc":
        df = pd.read_excel('filtered_country_names_kr.xlsx') 
        candidate_nes= df['name'].tolist()
    elif  ne_type =="per":
        df = pd.read_excel('update_korean_names.xlsx') 
        candidate_nes = df['name'].tolist()
    
    mentioned_nes = {} # key: country, value: mentioned sentences in list form
    _,sentences = split_text_label(DATASET)
    for sentence in sentences:
        for ne in candidate_nes:
            if ne in sentence:
                #print(country, " -- ", sentence)
                if ne not in mentioned_nes:
                    mentioned_nes[ne]=[]
                mentioned_nes[ne].append(sentence)

    #find the countries that are mentioned several times
    to_be_replaced={} #key:country, value: sentences to use for augmentation 
    ne_cnt={key: len(value) for (key, value) in mentioned_nes.items()}

    if ne_type =="loc":
        for country, cnt in ne_cnt.items():
            if cnt>2:
                if "조선" not in country and "대한민국" not in country: #exclude 조선인민주의공화국 and 대한민국
                    to_be_replaced[country]=mentioned_nes[country]
    elif ne_type =="per":
        for country, cnt in ne_cnt.items():
            to_be_replaced[country]=mentioned_nes[country]
    
    return to_be_replaced, candidate_nes



def create_augmented_data(DATA_DIR,ne_type):
    DATASET = os.path.join(DATA_DIR,"ko_train.conll")

    to_be_replaced,candidate_nes = get_to_replace_NE(DATASET,ne_type)
    augmented_data= {} #key: sentenceid, value: [['루마니아','B-LOC'],[],..] of augmented data
    original = {} #key: sentenceid, value: [['미국','B-LOC']] of training data 
    
    tags=["B-LOC","I-LOC"] if ne_type=="loc" else ["B-PER","I-PER"]
    with open(DATASET,'r', encoding='utf-8') as f:
        key = None
        for line in f:
            if line.startswith("# id") is True:
                key = line.replace("\n","") #update the key
                original[key]=[]
            elif len(line)>0 and line!="\n":
                line =  line.strip()
                fields = line.split()
                if fields[0] in to_be_replaced and fields[-1] in tags: #duplicate mentioned country AND is a location tag
                    if key not in augmented_data:
                        augmented_data[key]=[]
                original[key].append([fields[0],fields[-1]])

    text,_ = split_text_label(DATASET)
    for key in augmented_data: 
        #for sentence to augment
        split = text[key]
        new = []
        for idx,s in enumerate(split):
            replaced_nes = s
            while s in replaced_nes:
                replaced_nes= random.choices(candidate_nes,k=3)
            new.append(replaced_nes) #randomly selected country
            augmented_data[key].append([replaced_nes,original[key][idx][1]])
    return original,augmented_data

def write_augmented_file(DATA_DIR):
    """write the augmented data with original data as a new training data"""
    #get the augmented data 
    original, loc_augmented_data = create_augmented_data(DATA_DIR,"loc")
    _, per_augmented_data = create_augmented_data(DATA_DIR,"per")
    augmented_data = loc_augmented_data | per_augmented_data

    OUTPUT= os.path.join(DATA_DIR,"ko_augmented.conll")
    with open(OUTPUT,'w',encoding='utf8') as f:
        for key,value in augmented_data.items():
            for idx in range(3): #augment 3 sentences for each sentence
                f.write(key+"_augmented_"+str(idx)+"\n")
                #for each token in the sentence 
                for v in value:
                    if type(v[0]) == list:    #country to replace 
                        f.write(v[0][idx]+" _ _ "+v[1]+"\n")
                    else:
                        f.write(v[0]+" _ _ "+v[1]+"\n")
                f.write("\n")
       
        #write original training data 
        for key,value in original.items():
            f.write(key+"\n")
            for v in value:
                f.write(v[0]+" _ _ "+v[1]+"\n")
            f.write("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                      default=None,
                      type=str,
                      required=True,
                      help="The directory containing training data")
    args = parser.parse_args()

    write_augmented_file(args.data_dir)


if __name__ == '__main__':
  main()
