# simplifying-transformers
Deep Learning Course Project, ETH Zurich 2022


## retraining an Attention Block

To retrain a single Block using Bert as Teacher, run the following command:

```shell
# Retrain Block 6 (0 indexing, so it's the 7th Block in BERT), using MSE as criterion.
python .\retrain_single_block.py --block 6 --criterion MSE

# this command will retrain the Blocks by using the CrossEntropyLoss to compare the logits after the complete Attention Block 6
python .\retrain_single_block.py --block 6 --criterion CrossEntropy --loss_on_logits

# this command will retrain the Blocks by using the CrossEntropyLoss to compare the attention scores within the Attention Block 6
python .\retrain_single_block.py --block 6 --criterion CrossEntropy --loss_on_scores
```

After Running this Command for all 12 Blocks we have 12 Checkpoints of retrained Blocks.
notebooks/build_bert_from_retrained_12.ipynb will combine them into a runnable Bert where the Attention weights are obtained from the 12 retrained Blocks.

Afterwards running the following command will evaluate the Retrained and Teacher Model against Bert Large from Transformers (Huggingface)

```shell
# This will use 50% of the Test Data to run the Evaluation
python .\eval_merged_12.py --percentage_data 0.5
```

Note that it is important that the Jupyter book was run before to store a checkpoint containing the retrained Bert's weights.

## merge and train two layers

To merge two layers and then train the layer, run the following command:

```shell
# Merge layer 6 and 7(0 indexing, so it's the 7th and 8th layer in BERT), 
# and then train the new single layer with 12 heads with query, key and value of size 64 per head.
python .\retrain_merge.py --block 6 --block_heads 12 --block_d_k 64
```

To evaluate a merged layer, run the following command:

```shell
#Evaluate the layer we merged above
python .\eval_merge_layer.py --block 6 --block_heads 12 --block_d_k 64
```

To fine tune a model with 6 merge layers. First run the train layer for all even block so for block 0,2,4,6,8,10
Then name the files block 0, block 2 ... and put them in a folder namend merge_models_best 
Then run the following command:
```shell
#python .\


