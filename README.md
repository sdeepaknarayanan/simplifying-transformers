# Simplifying Transformers
Deep Learning Course Project, ETH Zurich 2022


We use ```Python 3.8.5``` for all our experiments.

```shell
conda create --name SiTra python=3.8

conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

conda install -c conda-forge transformers

conda install -c conda-forge tensorflow
```

To run the code on the euler cluster we provide examples in the .train and .eval bash scripts present in this repository.
The --storage_directory flag needs to be adapted to your storage directory.

Simply navigate to the codes source folder on euler and then run
```shell
env2lmod
bash .train
```

## Configuration

- The script has many command line arguments, all of which have defaults, but here is a list of important ones to run select experiments:
  - ```block``` to set the block you want to retrain: ```0 to 11```
  - ```block_d_k``` to set the effective dimensionality of the block to be retrained. Example Value: ```64```
  - ```block_heads``` to set the number of heads. Example Value: ```12```
  - ```block_dropout``` to set the dropout probability at the end of the attention block. Example Value: ```0.1```
  - ```batch_size``` - to set the batch size for training. Literature suggests higher values to average gradients. Example Value: ```64```
  - ```epochs``` - to set the number of epochs to train the model. Example Value: ```200``` (For Merge Experiments -- Train for 200 epochs, for Block Experiments, train for 50 epochs)
  - ```lr``` - to set the initial learning rate for training. Example Value: ```1e-3```
  - ```adam_weight_decay``` - to set the weight decay parameter of adam optim. Example Value: ```0.01```
  - ```adam_beta1``` - to set the beta1 parameter of adam. Example Value: ```0.9```
  - ```adam_beta2``` - to set the beta2 parameter of adam. Example Value: ```0.999``` for using pre-trained weights.
  - ```save_checkpoint_every``` - to indicate how often a permanent checkpoint should be written. Example Value: ```5``` Meaning every 5 epochs a checkpoint is stored on disk
  
  The example parameters mentioned here are the default hyperparameters and we use these hyperparameters generally. For finetuning we use a lower learning rate of ```1e-4```

### Checkpoints
Download the checkpoints from Polybox and put them into the folder /models/_checkpoints/wikitext2/

https://polybox.ethz.ch/index.php/s/kYrZwh47E8HUIT2

## Examples

### Retraining an Attention Block

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
python .\eval_retrained_12.py --percentage_data 0.5
```

To train a model by computing a loss over the output probabilities run
```shell
python train_block_till_end.py --block 4
```

To finetune the 12 retrained Layers run
```shell
python train_full_merged_model.py
```

Note that it is important that the Jupyter book was run before to store a checkpoint containing the retrained Bert's weights.

### Merge and Train multiple layers
---

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

To fine tune a model with 6 merge layers. First run the train layer for all even block so for block 0,2,4,6,8,10.

Then name the files block 0, block 2 ... and put them in a folder namend merge_models_best. 

Then run the following command:
```shell
#Fine tune all layers merged
python .\finetune_fully_merged.py
```

To fine tune the model conditionally, create the same folder with the merged layers as above.

Then run the following command:

```shell
#Fine tune all layers merged conditionally
python .\finetune_conditioned_fully_merged.py
```


To run evaluate on a completly merged model, so 6 merged layers. One must creat a folder with the merged layers the exact same way as for finetuning the merged layers.

Then run the following command:

```shell
#Evaluate a fully merged model
python .\evaluate_fully_merged_layer.py
```



