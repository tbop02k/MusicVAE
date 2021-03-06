# MusicVAE


### Introduction

* Only for Drum 

* Some hyper parameters and minor functionalities are different from original paper



### Reference

* [https://arxiv.org/pdf/](https://mailtrack.io/trace/link/7233bb35ccd13b3fe97c5e0cb90825a3bf1f974b?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1803.05428.pdf&userId=5832393&signature=388911f56318a357)[1803.05428.pdf](https://mailtrack.io/trace/link/7233bb35ccd13b3fe97c5e0cb90825a3bf1f974b?url=https%3A%2F%2Farxiv.org%2Fpdf%2F1803.05428.pdf&userId=5832393&signature=388911f56318a357)
* Dataset : [https://magenta.](https://mailtrack.io/trace/link/5b0d6601546cb1cb86bec0dfbbd63983c5ed7b93?url=https%3A%2F%2Fmagenta.tensorflow.org%2Fdatasets%2Fgroove&userId=5832393&signature=4a7e82253221a4c2)[tensorflow.org/datasets/groove](https://mailtrack.io/trace/link/5b0d6601546cb1cb86bec0dfbbd63983c5ed7b93?url=https%3A%2F%2Fmagenta.tensorflow.org%2Fdatasets%2Fgroove&userId=5832393&signature=4a7e82253221a4c2)

### Model

* based on paper (some hyper-parameters are different)
* non - teacher forcing (teacher forcing method will be added)



### Preprocess

```shell
python preprocess.py
```

* creates preprocessed pickle file

* you can define detail configurations in **config.py** 

### Training

```shell
python trainer.py
```

* creates model check point file

* you can define detail configurations in **config.py** and **model.py**

### Generation (makes midi file)

```shell
python generator.py
```

* creates .midi file by using model check point file



### Environments

- Python 3.7.6

- Pytorch 1.10

- PrettyMidi 0.2.9

- Cuda 10.2



