# SDETM

The SDETM is an extension of the Dynamic Embedded Topic Model (Dieng et al., arXiv 2019)  to corpora to model the topic evolution with temporal dependencies and predict document label simultaneously. 

[TODO]

Same as in DETM, the SDETM models each word with a categorical distribution whose parameter is given by the inner product between the word embedding and an embedding representation of its assigned topic at a particular time step. The word embeddings allow the SDETM to generalize to rare words. The SDETM learns smooth topic trajectories by defining a random walk prior over the embeddings of the topics. The SDETM is fit using structured amortized variational inference with LSTMs.

In contrast to DETM, SDETM also trains a linear classifier to predict document labels using the topic mixture. Hence, the model updates the topic embeddings to learn better not only the representation of the documents but also the label prediction of the documents.

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

## Example

To run the SDETM on the WHO dataset you can run the command below. You can specify different values for other arguments, peek at the arguments list in main.py.

```
python main.py --dataset who --data_path PATH_TO_DATA --emb_path PATH_TO_EMBEDDINGS --min_df 10 --num_topics 50 --lr 0.0001 --epochs 1000 --mode train
```


## Citation
```
@article{dieng2019dynamic,
  title={The Dynamic Embedded Topic Model},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={arXiv preprint arXiv:1907.05545},
  year={2019}
}
```


