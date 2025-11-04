# Three-Way Text Classifier by Lauren Palega 

## Background Information
**Text Classification** is a **machine learning** task where a model learns to assign predefined categories to text data based on patterns in the words. Categories can be anything: topics (sports, politics, science), product types, etc.

**Sentiment Analysis** is a type of **text classification** that identifies the sentiment expressed in text, such as positive, negative, or neutral.

In this project, we implement a **three-way text classifier** with optional **stop word removal** and support for **English and Spanish**. The classifier is a **machine learning model** because it **learns patterns from example text data** (the training files for each category) and then **predicts categories for new, unseen text**. It transforms text into numerical probabilities and applies the **Naive Bayes algorithm**, a probabilistic learning method, to assign the most likely category based on the words in the text.

- **Static test (`TestProgram.py`)** – Runs batch classification on predefined datasets and compares model accuracy with and without stop-word removal using small category files (≈10 lines each) to demonstrate improved model performance when stop words are excluded.  
- **Dynamic test (`TestProgramDynamic.py`)** – Interactive simulation that removes stop words and classifies user-provided sentences in real time using larger datasets (≈40 lines each).
---

## How Words Are Processed
### 1. Tokenization
Text is split into tokens (for ex. words).  

### 2. Stop Words Removal
Common words (e.g., "with", "and", "the") are ignored to focus on meaningful content.  
- Uses NLTK's stopwords list (`stopwords.words('english')` or `'spanish'`).  

### 3. Filtering Non-Alphabetic Tokens
Only alphabetic tokens are retained, removing numbers and punctuation.  

### 4. Frequency-Based Processing
- **Word Frequency Calculation:**  
Count occurrences of each word in every category: frequency[word] = [count_in_category1, count_in_category2, ...]
- **Word Probability Calculation:**  
Conditional probabilities for each word in each category are computed with Laplace smoothing to avoid zero probabilities(stored in a dictionary)
P(word | category) = (count(word in category) + 1) / (total words in category + unique words)
These probabilities represent how likely a word is to appear in a category.

### 5. Combining Probabilities (Naive Assumption) -> the classify method
Once we have the probability of each word in each category (from the training data), we want to calculate the probability that a new text belongs to each category
- We already calculated P(word | category) for each word from training data -> probability of being in each category 
- For a new text
    - We split it into tokens and check which words we have probabilities for 
    - For each !category!, we multiply the probabilities of all words in the text. 
        - We assume each word contributes independently to the probability of the category ("naive")
        - You initialize each category’s “score” to 1 instead of 0 because you’ll be multiplying probabilities
    - Highest combined probability is chosen as the category, unless difference is too small then itll be nuetral  

---

## Testing and Observations
I tested the classifier on two examples in both English and Spanish with `stop_words=True` and `stop_words=False`:
- Excluding stop words significantly improved the algorithm's accuracy on unseen messages.
- Including stop words sometimes caused the model to return neutral results instead of a clear category.
- For both languages, excluding stop words allowed the classifier to categorize messages correctly with higher probability.
**Conclusion:** Stop word removal improves text classification performance in this scenario.
