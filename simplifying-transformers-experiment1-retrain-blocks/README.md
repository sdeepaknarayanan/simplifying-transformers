# simplifying-transformers
Deep Learning Course Project, ETH Zurich 2022


## retraining an Attention Block

To retrain a single Block using Bert as Teacher, run the following Command

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

Afterwards running the following Command will evaluate the Retrained and Teacher Model against Bert Large from Transformers (Huggingface)

```shell
# This will use 50% of the Test Data to run the Evaluation
python .\eval_merged_12.py --percentage_data 0.5
```

Note that it is important that the Jupyter book was run before to store a checkpoint containing the retrained Bert's weights.
