# TUTORIAL: Sentiment analysis on Tweets using Hugging Face

The aim of this tutorial is to use NLP to analyse sentiments on Tweets.

**USE CASE**: all OVHcloud Tweets posted on *October 16, 2021*, i.e. 1 day after the company's IPO and 3 days after the incident.

The NLP allows us to show the Tweets sentiments according to their topic.

We will compare 3 different [Hugging Face](https://huggingface.co/) models:
- model based on [camemBERT](https://huggingface.co/transformers/model_doc/camembert.html): [pt-tblard-tf-allocine](https://huggingface.co/philschmid/pt-tblard-tf-allocine)
- model based on [BARThez](https://huggingface.co/transformers/model_doc/barthez.html): [barthez-sentiment-classification](https://huggingface.co/moussaKam/barthez)
- model based on [BERT](https://huggingface.co/transformers/model_doc/bert.html): [bert-base-multilingual-uncased-sentiment](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

## Sentiment Analysis

### Sentiment Analysis with pt-tblard-tf-allocine

*Théophile Blard, **French sentiment analysis with BERT**, (2020), [GitHub repository](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert)*

Tweets are divided into 2 classes according to their sentiment: **positive** or **negative**.

![camemBERT_results](https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/text-classification/hugging-face/sentiment-analysis-twitter/CamemBERT/results-camembert.png)

### Sentiment Analysis with barthez-sentiment-classification

*Eddine, Moussa Kamal and Tixier, Antoine J-P and Vazirgiannis, Michalis, **BARThez: a Skilled Pretrained French Sequence-to-Sequence Model**, (2020), [GitHub repository](https://github.com/moussaKam/BARThez)*

Tweets are divided into 2 classes according to their sentiment: **positive** or **negative**.

![BARThez_results](https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/text-classification/hugging-face/sentiment-analysis-twitter/BARThez/results-barthez.png)

### Sentiment Analysis with bert-base-multilingual-uncased-sentiment

*Refer to [NLP Town](https://www.nlp.town/)*

Tweets are divided into 5 classes, from 1 to 5 stars, according to their sentiment: 1 star corresponds to a **very negative** tweet while 5 stars corresponds to a **very positive** tweet.

![BERT_results](https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/text-classification/hugging-face/sentiment-analysis-twitter/BERT/results-bert.png)

## Comparison of models performance

Previously, we have tested 3 Hugging Face models based on BARThez, BERT and camemBERT. Two of them can be compared on our dataset: **BARThez** and **CamemBERT**.

It is possible to **process our data manually** and **compare our results** with the predictions of the models. Then, we will be able to display the success rate of the models to see which one was the best on our dataset.

The confusion matrix will also give us information about false positives or false negatives.

### Consufion matrix - BARThez x reel sentiments

![BARThez_matrix](https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/text-classification/hugging-face/sentiment-analysis-twitter/BARThez/confusion-matrix-barthez.png)

Success rate: 87.02 %

### Consufion matrix - CamemBERT x reel sentiments

![CamemBERT_matrix](https://github.com/ovh/ai-training-examples/blob/main/notebooks/natural-language-processing/text-classification/hugging-face/sentiment-analysis-twitter/CamemBERT/confusion-matrix-camembert.png)

Success rate: 78.63 %
