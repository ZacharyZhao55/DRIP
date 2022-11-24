# DRIP

env install:

    conda create -n drip python=3.9

    conda activate drip

    pip install --user stanfordcorenlp sentence-transformers==2.2.2 allennlp allennlp-models

    pip install torch==1.11.0


**run**:



train model(batch=256,epochs=30):
    train_data:'Data/train/bertTrain.tsv.gz','Data/train/bertTest.tsv.gz''

    python train_nil_v3_multi_task.py

model path: '/output/training_nli_multi-task_v3_(30)\_(batch_128)bert-base-uncased-(date)'

drip:
test_data path:
test on doucment:

    python req_segmenation(DRIP).py

    python req_segmenation(DRIP)-su.py

test on paragraph complexity:
        
    python req_segmenation(DRIP)-paragraphComplexity.py

    python req_segmenation(DRIP)-su-paragraphComplexity.py

download model:
    
