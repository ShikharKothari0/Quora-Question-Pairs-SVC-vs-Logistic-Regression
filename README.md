# Duplicate Question Detection

A machine learning project to classify whether two questions are duplicates using NLP-based text representations.

The project uses the Quora question-pairs dataset and explores how **Word2Vec embeddings** can be used to represent questions before applying traditional machine learning classifiers.

## Approach

1. Load and inspect the question-pair dataset.
2. Sample 50,000 question pairs for experimentation.
3. Clean the question text by:

   * Converting text to lowercase
   * Removing punctuation
   * Removing extra whitespace
   * Handling missing values
4. Tokenize the questions into individual words.
5. Train a **Word2Vec** model on the question corpus.
6. Represent each question by taking the mean of its word embeddings.
7. Concatenate the embeddings of the two questions to create the feature vector.
8. Train and compare classification models.

## Models

* **Logistic Regression**
* **Support Vector Machine (RBF kernel)**

## Results

| Model               |   Accuracy |
| ------------------- | ---------: |
| Logistic Regression |     72.39% |
| SVM (RBF)           | **77.23%** |

The SVM performed better than Logistic Regression on the test set.

## NLP Techniques

* Text preprocessing
* Tokenization
* Word2Vec word embeddings
* Sentence-level representation using averaged word vectors

## Machine Learning Concepts

* Binary classification
* Feature engineering
* Train/test split
* Support Vector Machines
* Logistic Regression
* Precision, recall and F1-score

## Dataset

The original dataset contains 404,290 question pairs. A random sample of 50,000 pairs was used for this experiment.

Each pair contains two questions and a binary label indicating whether they are duplicates.

## Technologies

* Python
* NumPy
* Pandas
* Gensim
* Scikit-learn
* Jupyter Notebook

## Limitations

The question representation is created by averaging Word2Vec embeddings. This provides a simple way to obtain fixed-length representations, but it does not capture word order or contextual meaning.

The project can be extended by experimenting with other text representations and more advanced models.
