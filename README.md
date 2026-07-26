# Visual Product Recommendation Engine

### A Deep Learning System for Content-Based Fashion Image Retrieval

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Dataset](#3-dataset)
4. [System Architecture & Approach](#4-system-architecture--approach)
5. [Implementation Walkthrough](#5-implementation-walkthrough)
6. [Model Design: Siamese Network & Triplet Loss](#6-model-design-siamese-network--triplet-loss)
7. [Training Process](#7-training-process)
8. [Similarity Search with FAISS](#8-similarity-search-with-faiss)
9. [Evaluation](#9-evaluation)
10. [Deployment: Streamlit Web App](#10-deployment-streamlit-web-app)
11. [Results & Discussion](#11-results--discussion)
12. [Limitations](#12-limitations)
13. [Future Work](#13-future-work)
14. [Project Structure](#14-project-structure)
15. [Setup & Installation](#15-setup--installation)
16. [How to Run](#16-how-to-run)
17. [Tech Stack Summary](#17-tech-stack-summary)
18. [References & Acknowledgements](#18-references--acknowledgements)

---

## 1. Introduction

Online fashion retail has grown into one of the largest and most visually driven segments of e-commerce. Customers rarely search for products the way they search for text documents — they search with their eyes. A shopper who sees a jacket they like on the street, in a magazine, or on social media wants to find "more things that look like this," not necessarily items tagged with the right keywords. Traditional keyword- or metadata-based search systems struggle here, because they depend on manually curated tags, category labels, and text descriptions that are often incomplete, inconsistent, or simply unavailable at scale.

This project addresses that gap by building a **Visual Product Recommendation Engine** — a system that takes a fashion product image as input and returns other products from a catalog that are visually similar, using only the pixel content of the images. No text search, no manual tagging, and no keyword matching is required. The system learns a numerical "fingerprint" (an embedding) for every image such that visually and stylistically similar products end up close together in a mathematical embedding space, while dissimilar products end up far apart.

The project was built and iterated inside a single Jupyter/Colab notebook, `Final_project_of_celebal.ipynb`, and progresses in stages: starting from an off-the-shelf pretrained convolutional network, moving through a purpose-built triplet-based training pipeline, and ending with a simple deployable web application. This README documents the full pipeline, the reasoning behind each design decision, how to reproduce it, and directions for extending it further.

## 2. Problem Statement

Given a catalog of fashion product images (shirts, shoes, watches, handbags, etc.) and a query image supplied by a user, the goal is to:

1. Represent every image in the catalog as a fixed-length numerical vector (an **embedding**) that captures its visual and stylistic characteristics.
2. Represent the query image using the same embedding function.
3. Retrieve the top-**K** catalog images whose embeddings are closest to the query embedding, using a similarity metric (cosine similarity).
4. Ensure the retrieved images are genuinely relevant — i.e., visually and categorically similar to the query — which requires the embedding space to be *discriminative*, not just generically descriptive.

A generic pretrained image classifier (like ResNet50 trained on ImageNet) produces embeddings that capture broad visual concepts, but it was never trained to understand that "these two t-shirts are more similar to each other than either is to a shoe." To make the embedding space genuinely useful for **product recommendation**, the model needs to be fine-tuned specifically for **similarity learning**, which is the central technical challenge this project tackles.

## 3. Dataset

The project uses the **Fashion Product Images (Small)** dataset, hosted on Kaggle by Param Aggarwal. This is a widely used benchmark dataset for fashion visual search and recommendation tasks.

**Key characteristics:**

- Thousands of product images spanning categories such as topwear, bottomwear, footwear, watches, bags, and accessories.
- Each image is accompanied by structured metadata in `styles.csv`, including fields such as `id`, `gender`, `masterCategory`, `subCategory`, `articleType`, `baseColour`, `season`, and `usage`.
- Images are provided as individual `.jpg` files, named according to their product `id` (e.g., `10001.jpg`), which allows direct mapping between metadata rows and image files.
- The dataset is downloaded programmatically via the Kaggle API and unzipped locally into a `dataset/` directory during the notebook's setup phase.

**Data selection strategy used in the project:**

Rather than training on the entire dataset (which would be computationally expensive for a single training run), the notebook constructs a **balanced subset**:

- The top 5 most frequent `articleType` categories are identified using `value_counts().nlargest(5)`.
- Up to 250 images per category are sampled, producing a balanced subset of at most 1,250 images.
- Each sampled row is mapped back to its corresponding image file, and any missing files are filtered out, producing two parallel arrays: `filtered_image_paths` and `filtered_labels`.

This balanced, moderately sized subset keeps training times reasonable on a single Colab GPU/CPU session while still providing enough intra-category and inter-category variety to teach the network meaningful distinctions.

## 4. System Architecture & Approach

The project follows a **two-phase strategy**, which is a common and effective pattern for building visual similarity systems on a limited compute budget:

### Phase 1 — Baseline retrieval with a frozen pretrained network

A ResNet50 model pretrained on ImageNet (with the classification head removed and global average pooling applied) is used directly as a feature extractor. Every catalog image is converted into a 2048-dimensional vector. These vectors are indexed in FAISS, and nearest-neighbor search is used to retrieve visually similar images. This phase requires **no training** and establishes a quick baseline to validate that the retrieval pipeline (feature extraction → indexing → search → visualization) works end-to-end.

### Phase 2 — Fine-tuned retrieval with a Siamese network and triplet loss

The baseline embeddings are generic — they were optimized for 1000-class ImageNet classification, not for fashion similarity. To specialize them, the project builds a **Siamese network**: three copies of the same embedding model (sharing weights) that process an anchor image, a positive image (same category as the anchor), and a negative image (different category) simultaneously. The network is trained with a **triplet loss** function that explicitly pulls same-category embeddings together and pushes different-category embeddings apart. After training, the same embedding model is used to re-encode the entire catalog, and a new FAISS index is built on these fine-tuned, task-specific embeddings.

This two-phase design lets the project demonstrate — concretely and visually — *why* fine-tuning matters, by comparing baseline recommendations against fine-tuned recommendations for the same query image.

## 5. Implementation Walkthrough

The notebook is organized as a sequence of 21 cells. Below is a detailed walkthrough of what each stage does and why.

### 5.1 Environment & data setup
The Kaggle CLI is installed, credentials are configured, and the fashion dataset is downloaded and unzipped into a local `dataset/` folder. This step assumes the user uploads their own `kaggle.json` API token before running the notebook (required for authenticating with Kaggle).

### 5.2 Baseline feature extractor
A ResNet50 model is loaded with `include_top=False` and `pooling='avg'`, meaning the final classification layers are discarded and a global average pooling layer condenses the last convolutional feature maps into a single 2048-dimensional vector per image. A helper function `extract_features()` handles loading an image, resizing it to 224×224 (ResNet's expected input size), applying ImageNet-specific preprocessing (`preprocess_input`), and running inference.

### 5.3 First triplet generator
An initial utility function, `generate_triplets()`, is written to randomly sample (anchor, positive, negative) triplets from a set of labeled images. This early version is later superseded by a more exhaustive triplet generator (Section 5.8) but establishes the core sampling logic: pick two images from the same category for anchor/positive, and one image from a different category for the negative.

### 5.4 Initial Siamese model skeleton
A first-draft multi-input Keras model is defined, taking three 224×224×3 image inputs (anchor, positive, negative) and passing each through the same frozen base ResNet50 to produce three embeddings. This early skeleton is a proof of concept for the shared-weight architecture used later in the fully trainable version.

### 5.5 Baseline retrieval demonstration (FAISS)
Using a small sample of 500 images, embeddings are extracted with the frozen ResNet50 extractor and assembled into a NumPy array (`all_embeddings`). A query embedding is computed for one sample image, setting up the retrieval demo in the next step.

### 5.6 Baseline FAISS index & search
A FAISS `IndexFlatIP` (inner-product) index is built over the L2-normalized baseline embeddings — inner product on normalized vectors is mathematically equivalent to cosine similarity. The top-5 nearest neighbors to the query embedding are retrieved.

### 5.7 Baseline visualization
Matplotlib is used to display the query image alongside its top-5 retrieved neighbors, with similarity scores annotated above each result. This gives an immediate qualitative sense of how well the untrained embeddings perform.

### 5.8 Metadata loading
The project loads product metadata, first attempting to read a JSON metadata file, then reading `styles.csv` into a pandas DataFrame (`styles_df`), with a fallback that searches the `dataset/` directory tree if the expected path isn't found. This metadata is essential for the next step, since it provides the category labels needed for triplet construction.

### 5.9 Balanced subset construction
As described in Section 3, the top 5 `articleType` categories are selected, and up to 250 images per category are sampled to build `balanced_df`. Image file paths are validated against the actual files on disk, producing `filtered_image_paths` and `filtered_labels` — the exact dataset used for fine-tuning.

### 5.10 Full triplet generation
A more thorough triplet generator, `create_triplets()`, iterates over **every image** in the balanced subset (rather than sampling a fixed number of random triplets), using each as an anchor once. For each anchor, a random positive is drawn from the same category (excluding the anchor itself) and a random negative is drawn from a different category. This produces one triplet per image in the subset, giving the network exposure to the full dataset during training.

### 5.11 Triplet sanity-check visualization
`plot_triplets()` displays a handful of generated (anchor, positive, negative) triplets side by side, with the anchor and positive framed in green and the negative in red, to visually confirm the triplet logic is behaving correctly before spending compute on training.

### 5.12 Trainable embedding model & Siamese network
This is the core modeling step (detailed further in Section 6): a new ResNet50 backbone is loaded with `include_top=False`, and its last 15 layers are left trainable while earlier layers are frozen, allowing the network to adapt high-level features to the fashion domain without destroying general-purpose low-level features learned from ImageNet. A `GlobalAveragePooling2D` layer, a `Dense(256)` projection, and an L2-normalization layer form the embedding head. This embedding model is then wrapped three times (for anchor, positive, negative inputs) into the final trainable Siamese network.

### 5.13 Custom triplet-loss training step
A custom Keras `Model` subclass, `SiameseModel`, overrides `train_step()` to compute the triplet loss manually (see Section 6.2 for the math) and apply gradients only to the shared embedding model's trainable weights.

### 5.14 Data pipeline and training
A `tf.data` pipeline reads, decodes, resizes, and preprocesses each image in a triplet, batches the triplets (batch size 32), and prefetches for performance. The model is trained for 5 epochs as a baseline configuration using the Adam optimizer with a learning rate of 0.0001.

### 5.15 Fine-tuned retrieval & visualization
The fine-tuned embedding model is used to re-encode the entire balanced subset into 256-dimensional embeddings. A new FAISS index is built on these embeddings, and the same query image used earlier is searched again — this time producing recommendations that should reflect what the network has learned about category similarity.

### 5.16 Evaluation metrics
`calculate_metrics()` computes **Precision@K** and **Recall@K** for a given query by comparing the categories of the retrieved images against the true category of the query, using the `filtered_labels` array as ground truth.

### 5.17–5.18 Streamlit app scaffold & deployment
A Streamlit script (`app.py`) is written to disk using the `%%writefile` magic command, providing an image upload widget and a placeholder for connecting the trained model and FAISS index. `streamlit`, and `localtunnel` (via `npm`) are installed, and the app is launched with a public tunnel URL so it can be accessed outside the Colab environment.

## 6. Model Design: Siamese Network & Triplet Loss

### 6.1 Why a Siamese network?

A Siamese network consists of two or more identical sub-networks that share the same weights and are evaluated on different inputs simultaneously. In this project, the shared sub-network is the **embedding model**: a ResNet50-based feature extractor topped with a 256-dimensional dense projection and L2 normalization. Using shared weights guarantees that the anchor, positive, and negative images are all mapped into the *same* embedding space by the *same* function — which is essential, since the goal is to compare distances between embeddings meaningfully.

### 6.2 Triplet loss

For each training triplet `(A, P, N)` — anchor, positive, negative — the model computes embeddings `f(A)`, `f(P)`, `f(N)`. The triplet loss is defined as:

```
L = max( d(f(A), f(P)) − d(f(A), f(N)) + margin, 0 )
```

where `d(x, y)` is the squared Euclidean distance between two embeddings, and `margin` is a hyperparameter (set to 0.5 in this project) that enforces a minimum gap between the anchor–positive distance and the anchor–negative distance.

Intuitively:
- The first term, `d(f(A), f(P))`, should be small — the anchor and positive (same category) should be close together in embedding space.
- The second term, `d(f(A), f(N))`, should be large — the anchor and negative (different category) should be far apart.
- The `margin` prevents the network from taking a shortcut where it collapses everything to the same point (which would trivially satisfy "anchor-positive distance is small" but fail to be discriminative). By requiring the negative to be *further* than the positive by at least `margin`, the loss forces genuinely useful structure into the embedding space.
- The outer `max(..., 0)` means that once the margin condition is already satisfied for a given triplet, that triplet contributes zero loss — training focuses effort on the triplets that are still being confused.

### 6.3 Partial fine-tuning strategy

Rather than fine-tuning the entire ResNet50 backbone (which risks overfitting on a relatively small subset of ~1,250 images and is computationally expensive) or freezing it entirely (which would leave the embeddings generic), the project freezes all but the **last 15 layers** of ResNet50. This lets the network adapt its highest-level, most semantically abstract features to the fashion domain, while preserving the lower-level edge, texture, and color filters already well-learned from ImageNet pretraining.

## 7. Training Process

- **Optimizer:** Adam, learning rate = 0.0001
- **Loss:** Custom triplet loss with margin = 0.5
- **Epochs:** 5 (baseline configuration)
- **Batch size:** 32 triplets per batch
- **Input pipeline:** `tf.data.Dataset` built from parallel anchor/positive/negative file-path datasets, zipped together, mapped through a shared `preprocess_image()` function (JPEG decode → resize to 224×224 → ResNet50 preprocessing), batched, and prefetched with `AUTOTUNE` for pipeline efficiency.
- **Training data:** One triplet generated per image in the balanced subset (~1,250 triplets), meaning every image in the subset serves as an anchor exactly once per epoch.

Because the custom `SiameseModel.train_step()` only supervises the *relative* distances between embeddings (not absolute class labels), no `y` targets are needed — `model.fit(dataset, epochs=epochs)` is called directly on the triplet dataset.

## 8. Similarity Search with FAISS

**FAISS** (Facebook AI Similarity Search) is used as the nearest-neighbor search engine. The project uses `IndexFlatIP`, a simple flat (exhaustive) index based on inner product similarity. Two important preprocessing details make this equivalent to cosine similarity search:

1. All embedding vectors are L2-normalized (`faiss.normalize_L2`) before being added to the index.
2. The query vector is also L2-normalized before the search is issued.

Once normalized, the inner product between any two vectors equals the cosine of the angle between them, so `IndexFlatIP` effectively performs a cosine-similarity nearest-neighbor search. This is repeated twice in the notebook: once for the baseline (2048-dim) embeddings and once for the fine-tuned (256-dim) embeddings, allowing a direct before/after comparison.

`index.search(query, k)` returns both the similarity scores (`distances`) and the positional indices (`indices`) of the top-K matches, which are then mapped back to file paths for visualization.

## 9. Evaluation

To move beyond purely qualitative "does this look right" inspection, the project implements a quantitative evaluation using standard information-retrieval metrics:

- **Precision@K** = (number of retrieved items in the top-K that share the query's category) / K
- **Recall@K** = (number of retrieved items in the top-K that share the query's category) / (total number of items in the dataset that share the query's category, excluding the query itself)

These metrics are computed for a sample query (`calculate_metrics()`), using the ground-truth `articleType` labels from the balanced subset. Precision@K captures how "clean" the top results are, while Recall@K captures how much of the full pool of genuinely relevant items the system is able to surface within the top-K window.

## 10. Deployment: Streamlit Web App

The final stage of the project packages the recommendation engine into a lightweight web interface using **Streamlit**:

- `app.py` defines a simple UI: a title, description text, and a file uploader restricted to `.jpg`, `.jpeg`, and `.png` files.
- When a user uploads an image, it is displayed back to them, and the app currently shows a placeholder message indicating where the trained model and FAISS index should be loaded and queried.
- The app is launched inside the notebook with `!streamlit run app.py`, and made publicly accessible from outside the Colab VM using `npx localtunnel --port 8501`, with the Colab machine's public IP printed for reference (required by localtunnel's tunnel-verification step).

As shipped, the app is a **UI scaffold** — the model-loading and FAISS-query logic is intentionally left as a clearly marked extension point (see Section 12 and 13) rather than hard-coded, since persisting and reloading the trained artifacts requires saving them to disk first (not yet done in the notebook).

## 11. Results & Discussion

Running the notebook end-to-end produces two side-by-side recommendation panels for the same query image: one generated with the frozen, generic ResNet50 baseline embeddings, and one generated with the fine-tuned, triplet-trained embeddings. Qualitatively, the fine-tuned results are expected to show tighter category consistency — recommendations that more reliably match the query's actual `articleType` — because the embedding space has been explicitly shaped by category-aware triplet supervision, whereas the baseline space was only ever optimized for generic ImageNet classification.

The Precision@5 / Recall@5 metrics computed on the fine-tuned embeddings give a concrete, numerical measure of this improvement for the evaluated query, complementing the visual inspection. Because the project evaluates on a single example query by default, results are illustrative rather than statistically robust — Section 12 discusses this limitation, and Section 13 outlines how to extend the evaluation to the full subset or dataset for a more rigorous benchmark.

## 12. Limitations

- **Small-scale evaluation:** The Precision@K / Recall@K calculation is demonstrated on a single query image rather than averaged across many queries, so it should be treated as illustrative rather than a robust benchmark.
- **Limited training data and epochs:** Fine-tuning uses roughly 1,250 images across 5 categories for only 5 epochs — enough to demonstrate the approach, but likely below what's needed to reach a production-quality embedding space.
- **Random triplet sampling:** Positive and negative images are chosen randomly rather than via hard-negative mining, meaning many triplets may already be "easy" (zero loss) and contribute little learning signal.
- **No model/index persistence:** The trained embedding model, fine-tuned embeddings, and FAISS index are not saved to disk, so the Streamlit app cannot yet serve live predictions without additional work.
- **Single-machine, non-scalable indexing:** `IndexFlatIP` performs exhaustive search, which is fine for a few thousand images but would not scale efficiently to a catalog of millions of products without switching to an approximate nearest-neighbor index (e.g., FAISS's `IVF` or `HNSW` index types).

## 13. Future Work

- **Persist trained artifacts.** Save the trained `embedding_model` (e.g., `embedding_model.save('siamese_embedding_model.h5')`), the fine-tuned embeddings (`numpy.save` or pickle), and the FAISS index (`faiss.write_index`) so the Streamlit app can load them and perform genuine live search on uploaded images.
- **Hard-negative mining.** Instead of purely random negative sampling, select negatives that are currently *close* to the anchor in embedding space (i.e., "hard" negatives), which typically accelerates and improves triplet-loss training.
- **Scale up training.** Train for more epochs, unfreeze additional ResNet50 layers, and/or introduce a learning-rate schedule to squeeze out further gains.
- **Full-dataset evaluation.** Extend the balanced subset to the complete dataset (or a much larger sample) and compute Precision@K/Recall@K averaged over many query images for a statistically meaningful benchmark.
- **Approximate nearest-neighbor indexing.** Swap `IndexFlatIP` for a scalable FAISS index type (e.g., `IndexIVFFlat` or `IndexHNSWFlat`) to support larger catalogs with sub-linear search time.
- **Richer UI.** Add category/color/season filters, multiple recommendation "shelves," and price/availability metadata to the Streamlit app for a more realistic e-commerce experience.
- **Alternative backbones.** Experiment with more modern pretrained backbones (e.g., EfficientNet, ConvNeXt, or CLIP's vision encoder) as drop-in replacements for ResNet50.

## 14. Project Structure

```
.
├── Final_project_of_celebal.ipynb   # Main notebook — all code lives here
├── app.py                           # Streamlit app (written out via %%writefile)
├── dataset/                         # Downloaded & unzipped Kaggle dataset
│   ├── images/                      # Product images (id.jpg)
│   └── styles.csv                   # Product metadata
└── README.md                        # This file
```

## 15. Setup & Installation

### Prerequisites
- Python 3.8+
- A Kaggle account and API token (`kaggle.json`)
- Recommended: run in Google Colab for GPU access and easy `localtunnel` deployment

### Install dependencies
```bash
pip install tensorflow faiss-cpu streamlit kaggle pandas numpy matplotlib pillow
```

### Configure Kaggle credentials
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Download the dataset
```bash
kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip -q fashion-product-images-small.zip -d dataset
```

## 16. How to Run

1. Open `Final_project_of_celebal.ipynb` in Jupyter or Google Colab.
2. Run cells 1–7 to install dependencies, download data, and see the baseline (untrained) retrieval demo.
3. Run cells 8–15 to build the balanced subset, generate triplets, define the trainable Siamese network, and train it.
4. Run cells 16–17 to see fine-tuned recommendations and compute Precision@K / Recall@K.
5. Run cells 18–20 to write, launch, and tunnel the Streamlit app.
6. *(Optional, recommended extension)* Save the trained model and FAISS index, then update `app.py` to load them and perform live search on uploaded images.

## 17. Tech Stack Summary

| Layer | Tool / Library |
|---|---|
| Deep learning framework | TensorFlow / Keras |
| Backbone architecture | ResNet50 (ImageNet-pretrained) |
| Similarity learning | Custom Siamese network + Triplet Loss |
| Vector similarity search | FAISS (`faiss-cpu`) |
| Data wrangling | pandas, NumPy |
| Visualization | Matplotlib |
| Web app / demo UI | Streamlit |
| Public tunneling (Colab) | localtunnel (via npm) |
| Dataset source | Kaggle API |

## 18. References & Acknowledgements

- **Dataset:** Fashion Product Images (Small), by Param Aggarwal, hosted on Kaggle.
- **Backbone model:** ResNet50, as implemented in `tensorflow.keras.applications`, pretrained on ImageNet.
- **Similarity search:** FAISS (Facebook AI Similarity Search) library.
- **Triplet loss concept:** Based on the metric-learning approach popularized by FaceNet (Schroff et al.), adapted here for fashion product similarity instead of face verification.
- **Deployment:** Streamlit, for rapid interactive web app prototyping, and localtunnel, for exposing a local server publicly from a Colab session.

---

*This README was generated from the project notebook `Final_project_of_celebal.ipynb` to document its purpose, methodology, and usage for future reference and collaboration.*

