# Movie Recommendation System

## Overview

This repository contains a simple movie recommendation system built using the **Movie Dataset: Budgets, Genres, Insights** from [Kaggle](https://www.kaggle.com/datasets/utkarshx27/movies-dataset?resource=download). The dataset is relatively small, which limits the scope of advanced recommendation techniques like collaborative filtering. Instead, we focus on two content-based approaches:

1. **TF-IDF with Cosine Similarity** (Basic Method)
2. **Embedding-based Similarity using LangChain Models** (Advanced Method)

## Getting Started

### Fork and Clone the Repository

1. Fork this repository into your own GitHub account.
2. Clone it to your local machine:
   ```bash
   git clone https://github.com/Arcoding/lumaa-spring-2025-ai-ml.git
   cd lumaa-spring-2025-ai-ml
   ```

### Install Dependencies

Ensure you have Python installed and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Implementation Details

### 1. Load and Preprocess the Dataset

- Read the CSV file containing movie information.
- Clean and preprocess text-based columns like `overview`.
- Handle missing values appropriately.

### 2. Convert Text Data to Vectors

- **TF-IDF Approach:** Uses term frequency-inverse document frequency (TF-IDF) to transform text descriptions into numerical vectors.
- **Embedding-based Approach:** Uses a transformer-based model to generate dense embeddings for the movie overviews using LangChain's integration with various embedding model providers.

### 3. Compute Similarity and Generate Recommendations

#### **TF-IDF & Cosine Similarity**

- Convert movie overviews into TF-IDF vectors.
- Compute cosine similarity between the user’s input query and each movie.
- Return the top 3–5 most similar movies.

#### **Embedding-based Similarity**

- Uses LangChain's embedding models to convert each movie overview into a dense vector representation.
- Compute cosine similarity between the user query and movie embeddings.
- Since embeddings capture contextual meaning, this approach provides **better recommendations** than TF-IDF.

## What Are Embeddings?

Embedding models create a vector representation of a piece of text. LangChain integrates with various embedding model providers to allow seamless generation and usage of embeddings for tasks like similarity search and recommendations.

## Running the Code

To generate movie recommendations, run the script. If no argument is provided, the program will run a predefined example. However, users can provide an argument to generate recommendations based on their input. For example:

```bash
python submission.py "I love animated Japanese movies"
```
## Link to video explanation
[Google Drive](https://drive.google.com/file/d/1VfjDR8epca6v1g_4GowUObE5EpkySwon/view?usp=share_link)

## Recommendations and Future Improvements

- **Hard Filters for Better Recommendations:** In production systems, recommendations are often combined with hard filters to improve accuracy. Filtering by attributes such as language, genre, and other user preferences can refine the recommendations.
- **More Advanced NLP Models:** Explore GPT-based embeddings.
- **Efficient Similarity Search with Vector Databases:** In production, recommendations do not run on large datasets directly for every request. Instead, vector databases employ Approximate Nearest Neighbor (ANN) algorithms to efficiently retrieve similar movies in real time.

## Contribution

Feel free to fork the repository, improve the code, and submit a pull request!

---

**Note:** Ensure all required dependencies are installed before running the script.

