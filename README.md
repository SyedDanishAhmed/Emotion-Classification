# Emotion-Classification
Predict the class of sentiments from a predefined list of emotions.

**Objective**

The task is to predict the class of sentiments from a predefined list of emotions.

**Steps Followed**

1. Filter out all rows where the sentiment (‘context’) is not in the list of aforementioned list of sentiments i.e. {'sad', 'jealous', 'joyful', 'terrified'}
2. Synthesize the training attributes and labels i.e. ‘utterance’ as the attributes and ‘context’ as the label
3. Carry out data pre-processing steps like stop word removal, stemming, lemmatization
4. Normalize is Term Frequency Inverse Document Frequency (TF-IDF) using TfidfTransformer. This would normalize frequencies in a weighted fashion to a value between 0 and 1
4. Convert the utterances into a sparse bag-of-words (BOW) representation. Each cell value 1 should represent that word j is present in utterance i and value of 0 indicates that it is not, where j is the column index and i is the row index
5. Built two classifiers, first an SGD classifier for the utterance sentiment classification and performed error analysis on the train data
6. Secondly, built an MLP classifier using pre-trained word embeddings as the feature

**Dataset Link**

https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
