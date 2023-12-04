# Uma Abordagem de Classificação de Emoção em Músicas com Redes Neurais

Este repositório contém o código-fonte para reproduzir os resultados do trabalho desenvolvido como projeto final em uma disciplina sobre redes neurais artificiais. Neste trabalho, utilizando as informações do chromagrama, classificamos a emoção de uma amostra de música usando uma rede neural convolucional entre 5 classes.

## Dependências

```
pip install torch torchaudio librosa scikit-learn
```

**1. Criando Database**

```
$ python3 inf721_dataset.py
```

**1. Criando Modelo**

```
$ python3 inf721_model.py
```

**1. Treinando Modelo**

```
$ python3 inf721_train.py
```

**1. Inferências**

```
$ python3 inf721_inference.py
```
