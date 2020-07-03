# MixMedia

The MixMedia models the source-specific dynamic topic prior evolution with temporal dependencies and predict document label simultaneously. 

Same as in DETM (Dieng et al., axRiv 2019), MixMedia models each word with a categorical distribution whose parameter is given by the inner product between the word embedding and an embedding representation of its assigned topic. The word embeddings allow the MixMedia to generalize to rare words. 

The MixMedia learns smooth topic trajectories by defining a random walk prior. The MixMedia is fit using structured amortized variational inference with LSTMs.

In contrast to DETM, MixMedia models a set of source-specific topic dynamic prior. These prior determines the topic proportion mean for the documents from that source.

In contrast to DETM, MixMedia models a static set of the topic embedding, which is a more efficient and is also more suitable for modeling temporal documents with short time gap. Empirically, the topic analysis is also much easier with a fixed set of topics.

In contrast to DETM, MixMedia also trains a linear classifier to predict document labels using the topic mixture. Hence, the model updates the topic embeddings to learn better not only the representation of the documents but also the label prediction of the documents.

## Dependencies

+ python 3.6.7
+ pytorch 1.1.0

## Datasets

The pre-processed GPHIN and WHO datasets can be found below:

+ link here
+ link here

The pre-fitted embeddings can be found below:

+ link here

All the scripts to pre-process a dataset can be found in the folder 'scripts'. 

## Using LSTM for q_theta

Both one-hot encodings and word embeddings are allowed as input. Use `--one_hot_qtheta_emb` to specify whether to use one-hot encoding (default true). Other additional parameters include choice of architecture ('trm' for Transformer and 'lstm' for LSTM), number of layers, hidden size, number of attention heads (for Transformer only), dropout rate and whether using bidirectional LSTM (for LSTM only).

For now the pre-trained embeddings are by default stored in `fasttext_cache/crawl-300d-2M-subword.bin` under the same folder as the processing script (`data_preprocess.py`). This is not added to repository due to its size. If you need this, please contact Bruce.

## Example

To run the MixMedia on the WHO dataset you can run the command below. You can specify different values for other arguments, peek at the arguments list in main.py.

```
python main.py --dataset WHO --data_path PATH_TO_DATA --emb_path PATH_TO_EMBEDDINGS --min_df 10 --num_topics 50 --lr 0.0001 --epochs 1000 --mode train
```


## Citation
```
@article{li2020mixmedia,
  title={MixMedia},
  author={Li et al},
  journal={arXiv preprint},
  year={2020}
}
```


