# DRIP

env install:

    conda create -n drip python=3.9

    conda activate drip

    pip install --user stanfordcorenlp sentence-transformers==2.2.2 allennlp allennlp-models

    pip install torch==1.11.0
    
 put the stanford cnornlp model into /model/
 
 put the srl model into /model/
 
 corenlp and srl model download address: https://zenodo.org/record/7353667#.Y38hAHZBxD8


**run**:



train model(batch=256,epochs=30):
    train_data:'Data/train/bertTrain.tsv.gz','Data/train/bertTest.tsv.gz''

    python train_nil_v3_multi_task.py

model path: '/output/training_nli_multi-task_v3_(30)\_(batch_128)bert-base-uncased-(date)'

drip:

test on document:

    python req_segmenation(DRIP).py

    python req_segmenation(DRIP)-su.py

test on paragraph complexity:
        
    python req_segmenation(DRIP)-paragraphComplexity.py

drip-no-opt:

test on document:

    python req_segmenation(DRIP-no-opt).py

    python req_segmenation(DRIP-no-opt)-su.py

test on paragraph complexity:
        
    python req_segmenation(DRIP-no-opt)-paragraphComplexity.py

model download link:
    https://zenodo.org/record/7353667#.Y38hAHZBxD8
    
