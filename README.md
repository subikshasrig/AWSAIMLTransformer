# IMDB Sentiment Analysis Using a Custom Transformer

A Transformer-based deep learning model for binary sentiment classification of IMDB movie reviews.

## Project Overview

This project implements a custom Transformer architecture to classify movie reviews from the IMDB dataset as either **positive** or **negative**.

Instead of using a pre-trained BERT model for classification, the project uses the **BERT tokenizer** for text tokenization while building and training a custom Transformer model from scratch.

The model uses:

* Token embeddings
* Positional embeddings
* Multi-head self-attention
* Feed-forward neural networks
* Layer normalization
* Residual connections
* Mean pooling
* A binary classification layer

## Dataset

The project uses the **IMDB Large Movie Review Dataset**, containing:

* 25,000 labeled training reviews
* 25,000 labeled test reviews
* 12,500 positive and 12,500 negative reviews in each split

The original training set was divided into:

* **22,500 training samples**
* **2,500 validation samples**

The test set was kept separate for final evaluation.

## Model Architecture

The custom Transformer uses the following configuration:

| Parameter           |                               Value |
| ------------------- | ----------------------------------: |
| Vocabulary          | BERT `bert-base-uncased` vocabulary |
| Sequence length     |                                 128 |
| Embedding dimension |                                 128 |
| Transformer layers  |                                   4 |
| Attention heads     |                                   4 |
| Head size           |                                  32 |
| Dropout             |                                 0.1 |
| Number of classes   |                                   2 |
| Optimizer           |                               AdamW |
| Learning rate       |                                3e-4 |
| Epochs              |                                   3 |
| Batch size          |                                  32 |

Each Transformer block consists of:

1. Layer normalization
2. Multi-head self-attention
3. Residual connection
4. Feed-forward network
5. Residual connection

The final Transformer representations are mean-pooled across the sequence dimension and passed through a linear classification layer to produce two sentiment logits.

## Training Results

The model was trained for three epochs.

| Epoch   | Validation Accuracy |
| ------- | ------------------: |
| Initial |              49.76% |
| 1       |              70.76% |
| 2       |              76.24% |
| 3       |          **78.64%** |

The model achieved a final **75.98% accuracy on the unseen IMDB test set**.

### Final Result

**Test Accuracy: 75.98%**

This meets the project's target of achieving greater than 75% test accuracy.

## Project Structure

```text
IMDB-Transformer-Sentiment/
│
└── IMDB_Transformer_Sentiment.ipynb
```

The Jupyter Notebook contains the complete implementation, including dataset preparation, exploratory analysis, tokenization, Transformer architecture, training, validation, testing, and conclusions.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Pandas
* NumPy
* Matplotlib
* Jupyter / Google Colab

## How to Run

The notebook is designed to run in **Google Colab**.

1. Open `IMDB_Transformer_Sentiment.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. The notebook downloads and prepares the IMDB dataset.
4. The Transformer model is initialized and trained.
5. Validation accuracy is calculated after each epoch.
6. Final performance is evaluated on the test dataset.

## Key Takeaways

* Transformer architectures can be adapted for text classification rather than text generation.
* Multi-head self-attention allows the model to learn relationships between tokens in a review.
* Positional embeddings provide information about token order.
* Mean pooling converts token-level representations into a single review-level representation.
* The model improved from approximately random performance before training to **78.64% validation accuracy**.
* The final model achieved **75.98% test accuracy** on unseen reviews.

## Author

**Subiksha Sri**
