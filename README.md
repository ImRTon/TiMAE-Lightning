# Ti-MAE: Self-Supervised Masked Time Series Autoencoders
This is an unofficial implementation of Ti-MAE in PyTorch Lightning. The original paper can be found [here](https://arxiv.org/abs/2301.08871).  
There's also a experimental implementation of Ti-MAE using Mamba v2 instead of transformer, you can find more information about Mamba [here](https://github.com/state-spaces/mamba).  

> [!NOTE]  
> This is a work in progress.

## Installation
* Python >= 3.9  
* PyTorch  
* PyTorch Lightning  
* [mamba-ssm](https://github.com/state-spaces/mamba)  

## TODO
- Complete documentation
- Complete unit tests  
- Add CrossMAE to Ti-MAE  

## Training
Training is divided into two parts, pretraining and finetuning. Pretraining is done on the entire dataset without labels, while finetuning is done on the labeled dataset.  
In the pretraining phase, the model is trained to predict the masked values of the input data. The model is assembled with a transformer encoder and a transformer decoder. The encoder is used to encode the input data, while the decoder is used to predict the masked values.  
In the finetuning phase, the model is trained to predict the labels of the input data. The model is assembled with only a transformer encoder and a classifier. The encoder can be loaded with the pretrained weights from the pretraining phase.  

```bash
python train.py fit --model MODEL_NAME --model.lr LEARNINGRATE --model.input_dim INPUT_DATA_DIM \
    --model.num_heads NUM_HEADS --model.num_layers NUM_LAYERS --model.emb_size EMBEDDING_SIZE \
    --model.intermediate_dim INTERMEDIATE_DIM --model.cls_embed True --model.max_len DATA_MAX_LEN \
    --model.mask_ratio MASK_RATIO --model.seq_len DATA_SEQ_LEN --data.batch_size BATCH_SIZE \
    --data.dataset_path DATASET_PATH --data.pretrain PRETRAIN --data.normalize NORMALIZE \
    --data.class_id_start CLS_ID_START --data.crop_size DATA_SEQ_CROP_SIZE \
    --trainer.max_epochs MAX_EPOCHS --trainer.callbacks+=LearningRateMonitor \
    --trainer.callbacks.logging_interval=step --trainer.log_every_n_steps 1
```
* `MODEL_NAME`: The model to train, available models are `TiMAE`, `TiMAEForPretraining`, `TiMAEMamba`, `TiMAEMambaForPretraining`.  
* `LEARNINGRATE`: The learning rate for the model.  
* `INPUT_DATA_DIM`: The input data dimension, for UCR2018 dataset is `1`.  
* `NUM_HEADS`: The number of heads for the transformer.  
* `NUM_LAYERS`: The number of layers for the transformer.  
* `EMBEDDING_SIZE`: The embedding size for the transformer.  
* `INTERMEDIATE_DIM`: The intermediate dimension for the transformer.  
* `DATA_MAX_LEN`: The maximum sequence length of the input data for the model.  
* `MASK_RATIO`: The ratio of the masking strategy for the MAE model.  
* `DATA_SEQ_LEN`: The sequence length of the input data for the model when training.  
* `BATCH_SIZE`: The batch size for the model.  
* `DATASET_PATH`: The path to the dataset.tsv.  
* `PRETRAIN`: Whether to pretrain the model, it's used as an indicator for the datamodule wheather to pass the labels or not.  
* `NORMALIZE`: Whether to normalize the input data.  
* `CLS_ID_START`: The starting class id for the dataset, it can be automatically set by the datamodule.  
* `DATA_SEQ_CROP_SIZE`: The crop size of the input data for the model when training.  
* `MAX_EPOCHS`: The maximum number of epochs to train the model.  

Below are the example commands to train the model.

### Pretraining
```bash
python train.py fit --model TiMAEForPretraining --model.lr 1e-3 --model.input_dim 1 --model.num_heads 4 --model.num_decoder_heads 4 --model.num_layers 2 --model.num_decoder_layers 2 --model.emb_size 64 --model.decoder_emb_size 32 --model.intermediate_dim 256 --model.decoder_intermediate_dim 128 --model.cls_embed True --model.max_len 140 --data.batch_size 64 --data.dataset_path datasets/UCRArchive_2018/ECG5000/ECG5000_TEST.tsv --data.val_dataset_path datasets/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv --data.pretrain True --trainer.max_epochs 1000  --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --model.seq_len 140 --data.normalize False --trainer.log_every_n_steps 1 --model.mask_ratio 0.75
```
```bash
python train.py fit --model TiMAEMambaForPretraining --model.lr 1e-3 --model.input_dim 1 --model.num_heads 8 --model.num_decoder_heads 8 --model.ssm_state_size 64 --model.decoder_ssm_state_size 64 --model.emb_size 64 --model.decoder_emb_size 32 --model.cls_embed True --data.batch_size 64 --data.dataset_path datasets/UCRArchive_2018/ECG5000/ECG5000_TEST.tsv --data.val_dataset_path datasets/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv --data.pretrain True --trainer.max_epochs 1000  --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --model.seq_len 140 --data.normalize False --trainer.log_every_n_steps 1 --model.mask_ratio 0.75
```

### Finetuning
```bash
python train.py fit --model TiMAE --model.lr 1e-3 --model.input_dim 1 --model.num_heads 4 --model.num_layers 2 --model.emb_size 64 --model.intermediate_dim 256 --model.cls_embed True --model.max_len 140 --data.batch_size 64 --data.dataset_path datasets/UCRArchive_2018/ECG5000/ECG5000_TRAIN.tsv --data.pretrain False --trainer.max_epochs 20  --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --data.normalize False --trainer.log_every_n_steps 1 --model.encoder_ckpt_path lightning_logs/version_0/checkpoints/epoch\=999-step\=71000.ckpt
```

### Testing
```bash
python train.py test --model TiMAE --model.lr 1e-3 --model.input_dim 1 --model.num_heads 4 --model.num_layers 2 --model.emb_size 64 --model.intermediate_dim 256 --model.cls_embed True --model.max_len 140 --data.batch_size 64 --data.dataset_path datasets/UCRArchive_2018/ECG5000/ECG5000_TEST.tsv --data.pretrain False --trainer.max_epochs 100  --trainer.callbacks+=LearningRateMonitor --trainer.callbacks.logging_interval=step --data.normalize False --trainer.log_every_n_steps 1 --ckpt_path lightning_logs/version_1/checkpoints/epoch\=99-step\=700.ckpt
```

## Results

### UCR2018
| Dataset | Model       | Params | Method      | Accuracy | F1       |
| ------- | ----------- | ------ | ----------- | -------- | -------- |
| ECG5000 | TiMAE       | 91.7 K | CLS TOKEN   | 0.937556 |          |
| ECG5000 | TiMAE       | 91.7 K | AVG POOLING | 0.941556 |          |
| ECG5000 | TiMAE-Mamba | 35.7 K | AVG POOLING | 0.935778 | 0.611325 |
| ECG5000 | TiMAE-Mamba |  1.7 M | CLS TOKEN   | 0.927111 | 0.640663 |
| ECG5000 | TiMAE-Mamba |  1.7 M | AVG POOLING | 0.929778 | 0.620936 |

## References
* [Transformers ViTMAE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit_mae/modeling_vit_mae.py)  
* [Ti-MAE Paper](https://arxiv.org/abs/2301.08871)  
* [StatQuest Decoder Transformer from scratch](https://github.com/StatQuest/decoder_transformer_from_scratch)  