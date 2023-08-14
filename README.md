# BERT-email-classification

# Importing Libraries:

TensorFlow (tf): A popular deep learning framework used for building and training neural networks.
TensorFlow Hub (hub): A library for sharing, discovering, and reusing pre-trained model components (like BERT).
TensorFlow Text (text): An extension to TensorFlow for working with text data.
Pandas (pd): A library for data manipulation and analysis.

# Importing Dataset:

The provided dataset is loaded from a CSV file named "spam.csv" using the Pandas library.
The dataset contains two columns: "Category" (ham or spam label) and "Message" (email content).

# Data Exploration:

Basic exploration of the dataset is performed:
Grouping the data by the "Category" column to see the count, unique entries, top messages, and their frequencies for each category (ham or spam).
Counting the number of occurrences for each category using .value_counts().

# Balancing the Dataset:

To handle the class imbalance (more ham than spam messages), the code downsamples the "ham" class to match the number of instances in the "spam" class.
This balanced dataset is stored in df_balanced.

# Text Preprocessing and BERT Embedding:

Two BERT-related models are loaded using TensorFlow Hub:
bert_preprocess: A BERT model for text preprocessing, which includes tokenization and input formatting.
bert_encoder: The main BERT model for generating embeddings from preprocessed text.
The function get_sentence_embedding preprocesses sentences and uses the BERT encoder to get the pooled embedding vectors for these sentences.
# Building the Functional Model:

A functional API is used to create the classification model.
text_input is an input layer that accepts strings as inputs (email messages).
The input text is preprocessed using the bert_preprocess layer, and the resulting preprocessed text is fed into the bert_encoder to get BERT embeddings.
A dropout layer is applied to the pooled BERT output to prevent overfitting.
Finally, a dense layer with a sigmoid activation function outputs a single value, representing the predicted probability of being spam.

# Compiling the Model:

The model is compiled with the Adam optimizer, binary cross-entropy loss function (suitable for binary classification), and custom evaluation metrics (accuracy, precision, and recall).
# Training the Model:

The model is trained using the training data (X_train and y_train) for a specified number of epochs (10 in this case).
# Model Evaluation:

After training, the model's performance is evaluated using the test data (X_test and y_test).
The loss and the defined evaluation metrics (accuracy, precision, recall) are printed.
# Prediction and Inference:

The trained model is used to predict the classes of new email messages (reviews).
The output is an array of predicted probabilities indicating the likelihood of being spam.
# Confusion Matrix and Classification Report:

The code generates a confusion matrix using confusion_matrix from scikit-learn to visualize the model's performance on the test data.
The classification report is also printed, showing metrics like precision, recall, and F1-score for both classes (spam and ham).





