# Fake News Detection Using NLP and Fine Tune With BERT 

![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/a6ebc82e-2a36-4a44-aab0-3d62d26664ab)

# Introduction
This project focuses on the detection of fake news using Natural Language Processing (NLP) techniques and BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to build a machine learning model that can accurately classify news articles as either fake or true based on their textual content.

![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/50b817da-321a-419f-84bd-83ed60fbd074)

# Dataset
The project utilizes a dataset consisting of labeled news articles, with a combination of true and fake news samples 44000 rows and 3 columsn. The dataset is divided into a training set, validation set, and test set.After Feature Engg these are the Columns [title,text,subject,date,Target,label].The data is preprocessed and tokenized for further analysis and model training.

# Technologies Used
* Python
* Numpy
* Pandas
* PyCaret
* Transformers (Hugging Face)
* PyTorch
* Matplotlib
* Scikit-learn
  
# Methodology
**NOTE**: This Project is Implemented using Google Colab which is Optional
* Data Preprocessing: The dataset is loaded and preprocessed, including label generation and data merging.

* Exploratory Data Analysis: Visualizations are performed to gain insights into the dataset, such as the distribution of the target classes.
![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/e54efdcf-7c10-497a-9f7d-be57e610af66)

![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/5e1efe83-4d14-44ad-9f0b-80edc17cd8e2)

* Model Development: The BERT model is fine-tuned using the training set. The BERT tokenizer is utilized to tokenize the text data.

* Model Training: The dataset is converted into tensors and fed into the BERT model. The model parameters are frozen, and a custom classification layer is added.

* Model Evaluation: The trained model is evaluated using the validation set, and performance metrics such as precision, recall, and F1-score are calculated.
![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/cdaa9cb0-02c3-4ca3-8597-9f09d9e36175)

* Model Testing: The final model is tested on the test set to assess its generalization performance.

# Results
![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/95aac512-bf18-4591-9b1c-9be2c06ddc28)

The model achieves good performance in classifying between fake and true news articles, with high precision, recall, and F1-score for both classes. The classification report provides detailed metrics for the model's performance.

# Why Torch and BERT?
In this project, torch (PyTorch) and BERT (Bidirectional Encoder Representations from Transformers) have been utilized for several reasons:
**PyTorch (torch)**
* PyTorch is a popular open-source deep learning framework that provides a flexible and efficient platform for building and training neural networks.
It offers dynamic computation graphs, making it easier to define and modify network architectures.
* PyTorch provides extensive support for GPU acceleration, allowing for faster model training and evaluation.
The rich ecosystem of PyTorch includes various prebuilt models, optimization algorithms, and evaluation metrics, which are beneficial for this project.

**BERT (Bidirectional Encoder Representations from Transformers)**
* BERT is a state-of-the-art transformer-based model developed by Google, trained on a large corpus of text data.
It has achieved remarkable success in various NLP tasks, including text classification, question answering, and named entity recognition.
* BERT has a deep understanding of context and captures intricate relationships between words, resulting in better semantic representation of the text.
It utilizes the attention mechanism to capture contextual information from both left and right contexts, making it well-suited for understanding the meaning of words and sentences.

# Industrial Use Cases
BERT and similar transformer-based models have been widely adopted in various industrial applications, including:
* Fake News Detection: BERT can effectively distinguish between fake and true news articles by leveraging its contextual understanding and semantic representation capabilities.
* Sentiment Analysis: BERT can accurately analyze the sentiment of text, enabling businesses to understand customer opinions and feedback for product or service improvement.
* Question Answering Systems: BERT can be used to build intelligent question answering systems that provide accurate and relevant answers based on the given context.
* Text Summarization: BERT can generate concise and informative summaries of long documents, enabling users to quickly grasp the main points without reading the entire text.
* BERT's ability to capture contextual information and its strong performance in various NLP tasks make it a valuable tool for addressing real-world challenges in industries such as news media, e-commerce, customer support, and information retrieval systems.

By incorporating BERT into the fake news detection project, the model benefits from BERT's deep contextual understanding and semantic representation, leading to enhanced accuracy and robustness in classifying between fake and true news articles.

# Usage
Clone the repository: git clone https://github.com/your-Aftabbs/Fake-News-Detection.git
Install the required dependencies: pip install -r requirements.txt
Run the main script: python main.py
Modify the script as needed, such as adjusting hyperparameters or preprocessing steps.

# Future Enhancements
- Fine-tune the model with additional data to improve performance.
- Experiment with different NLP techniques or transformer-based models.
- Implement a web interface or API for real-time fake news detection.

# References
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- Hugging Face Transformers Documentation
- PyTorch Documentation

# Contributing
Contributions to this project are welcome. Feel free to open issues or submit pull requests with any improvements or bug fixes.

# THANK YOU
![image](https://github.com/Aftabbs/Fake-News-Detection-Using-NLP-and-BERT/assets/112916888/0ad00567-bf32-4535-8f11-a85d346c62b2)

