# semeval2022-task11
## UA-KO @ SemEval-2022 Task 11 : MultiCoNER

This is the code for the SemEval task 11 MultiCoNER (Multilingual Complex Named Entity Recognition) on Korean language data.

Quick Start

1. Install requirements
         
         python -m pip install -r requirements.txt
          
         
2. Train model 
         
         python train.py --data_dir data --pretrained_model kobert --max_seq_length 256 --output_dir output --train_batch_size 16 --num_train_epochs 1 
         

3. Evaluate model

         python eval.py --data_dir data --pretrained_model kobert --max_seq_length 256 --model_dir output/trained_model --output_dir output
         
         
         
