# Advanced NLP & Time Series Forecasting Projects

This repository contains a curated collection of projects demonstrating foundational and advanced techniques in Natural Language Processing (NLP) and Time Series Forecasting. This work is part of my personal data science portfolio.

The projects showcase a clear progression of skills:

1.  **Classical NLP:** Text classification using feature extraction (TF-IDF).
2.  **Unsupervised NLP:** Discovering hidden structures in text (LDA Topic Modeling).
3.  **Modern NLP:** Capturing semantic meaning with dense vectors (Word2Vec).
4.  **Time Series:** Forecasting data with complex seasonal patterns (Prophet).

-----

## Project 18: Spam SMS Detection

  * **Folder:** `Project_18_Spam_Detection/` (or similar)
  * **Objective:** To build a classic binary text classifier to distinguish spam messages ("spam") from legitimate messages ("ham").

### Dataset

[SMS Spam Collection Data Set](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle. A small, clean dataset perfect for demonstrating classification fundamentals.

### Key Techniques

  * **Text Preprocessing:** Implemented a full pipeline using `NLTK` to clean text by lowercasing, removing punctuation, removing stopwords, and applying stemming.
  * **Feature Extraction:** Compared two classic methods:
    1.  `CountVectorizer` (Bag-of-Words)
    2.  `TfidfVectorizer` (Term Frequency-Inverse Document Frequency)
  * **Modeling:** Used `Scikit-learn` to train a `MultinomialNB` (Multinomial Naive Bayes) classifier, which is highly effective for text classification.

### Analysis & Findings

This project served as a strong foundation in text preprocessing. The key insight was observing the performance difference between vectorization methods. The **TF-IDF vectorizer** model consistently outperformed the basic Bag-of-Words, as it correctly penalizes common words (like "the" or "is") that appear in all documents and boosts the importance of words that are rare but highly indicative of a specific class (like "congratulations" or "winner" in spam).

The final model achieved **\~98% accuracy**, demonstrating the power of TF-IDF and Naive Bayes for this type of sparse text data.

-----

## Project 20: Topic Modeling with LDA

  * **Folder:** `Project_20_Topic_Modeling/`
  * **Objective:** To apply unsupervised machine learning (**Latent Dirichlet Allocation - LDA**) to a corpus of news articles to automatically discover the main topics they discuss.

### Dataset

[BBC News Classification](https://www.kaggle.com/competitions/learn-ai-bbc/data) from Kaggle. This dataset was ideal because it includes labels (sport, tech, etc.) that were *not* used in training but were invaluable for validating the model's unsupervised results.

### Key Techniques

  * **Unsupervised NLP:** This project was a deliberate move from supervised to unsupervised learning.
  * **Advanced Preprocessing:** Used `NLTK`'s `WordNetLemmatizer` to ensure words like "running" and "ran" were grouped as "run," which is critical for topic coherence.
  * **`Gensim` Library:**
      * Created a `corpora.Dictionary` to map words to IDs.
      * Built a `Bag-of-Words (BoW) Corpus` as required by the model.
      * Trained the `LdaModel` to find 5 distinct topics.

### Analysis & Findings

The most insightful part of this project was validating the model's output. After training the LDA model on the *unlabeled* text, I examined the keywords for each of the 5 discovered topics.

  * **Topic 0:** `game`, `team`, `player`, `win`, `match`...
  * **Topic 1:** `government`, `labour`, `election`, `party`, `blair`...
  * **Topic 2:** `film`, `music`, `show`, `best`, `star`...

These topics clearly and accurately mapped to the dataset's *actual* hidden labels of **"Sport"**, **"Politics"**, and **"Entertainment"**. This proved the model's ability to find meaningful thematic structures in raw text without any human guidance.

-----

## Project 23: Time Series Forecasting with Prophet

  * **Folder:** `Project_23_Prophet_Forecasting/`
  * **Objective:** To build a robust time series forecast for daily bike rentals, capturing multiple seasonalities and the impact of holidays.

### Dataset

[Bike Sharing Dataset (Daily)](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset) from Kaggle. This dataset is perfect for Prophet as it contains strong, multi-level seasonality (weekly and yearly) and a clear holiday-related impact.

### Key Techniques

  * **`Prophet` Library:** Used Facebook's forecasting tool, which is designed to be robust and automated.
  * **Data Preparation:** Renamed columns to Prophet's required `ds` (datestamp) and `y` (target value) format.
  * **Holiday Modeling:** Created a custom `holidays` dataframe to pass to Prophet, allowing the model to learn the specific effect of public holidays on rentals.
  * **Visualization:** Relied heavily on `model.plot_components()` to understand the patterns the model isolated.

### Analysis & Findings

Prophet excels at "glass-box" forecasting, where the components are interpretable. The `plot_components` function was the key analytical tool, providing immediate insights:

1.  **Trend:** A clear, non-linear upward trend showing the growing popularity of the bike-sharing service.
2.  **Yearly Seasonality:** A strong, smooth wave peaking in the summer and bottoming out in winter, as expected.
3.  **Weekly Seasonality:** A distinct pattern showing that rentals were *higher* on weekdays and *lower* on weekends, suggesting the service was heavily used by commuters.
4.  **Holiday Effect:** By adding the custom holiday list, the model learned that holidays had a unique, negative impact on rentals, similar to a weekend.

This project demonstrated the power of modern, automated forecasting tools to handle complex, human-centric time series data with minimal manual tuning.

-----

## Project 24: Word Embeddings with Word2Vec

  * **Folder:** `Project_24_Word2Vec/`
  * **Objective:** To train a `Word2Vec` model to learn the *semantic meaning* of words by representing them as dense numerical vectors.

### Dataset

[BBC News Classification](https://www.kaggle.com/competitions/learn-ai-bbc/data) (re-used from Project 20). This dataset was chosen again because its concentrated, domain-specific vocabulary (politics, tech, etc.) allows the model to learn strong contextual relationships even with a relatively small corpus.

### Key Techniques

  * **`Gensim` Library:** Used `gensim.models.Word2Vec` to train the model.
  * **Corpus Preparation:** The input was a "list of lists," where each inner list was the tokenized text of a single news article.
  * **Vector Exploration:**
      * `model.wv.most_similar()`: To find the closest words (neighbors) in the vector space.
      * `model.wv.doesnt_match()`: To find the "odd one out" in a list.
      * **Vector Arithmetic:** To solve analogies (e.g., `A` is to `B` as `C` is to `?`).

### Analysis & Findings

This project was a fantastic introduction to modern NLP, moving beyond word counts to word *meaning*. After training, the model successfully captured the context of the news corpus.

  * **Word Similarity:** When asked for words similar to `government`, the model returned `labour`, `blair`, `election`, and `party`.
  * **Finding the Outlier:** In the list `['film', 'music', 'show', 'election']`, the model correctly identified `election` as the word that didn't belong.

The most famous test, `king - man + woman = queen`, was not successful, which itself was a key insight. It highlights that the model's "understanding" is **entirely dependent on its training data**. Since words like "king," "man," and "woman" were not common in a 2004-2005 news dataset, it couldn't learn that specific relationship. However, it successfully solved domain-specific analogies, proving that it had learned the semantic relationships *within its own world*.

-----

## Technologies Used

  * **Python 3.x**
  * **Data Manipulation:** `Pandas`, `NumPy`
  * **NLP:**
      * `NLTK`: For text preprocessing (tokenization, stopwords, stemming, lemmatization).
      * `Scikit-learn`: For `CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`, and model evaluation.
      * `Gensim`: For `LdaModel` (Topic Modeling) and `Word2Vec` (Embeddings).
  * **Time Series:** `Prophet`
  * **Plotting:** `Matplotlib`
  * **Environment:** `Jupyter Notebook`
