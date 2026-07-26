Visual Product Recommendation Engine 👗🔍

A deep learning-based visual search system for fashion products. Upload an image of a clothing item and the system retrieves visually similar products using image embeddings and similarity search — no text tags or metadata required.

Overview

This project builds an end-to-end content-based image retrieval (CBIR) pipeline for fashion e-commerce, going from a baseline pretrained feature extractor to a fine-tuned Siamese network trained with triplet loss, and finally wraps the model in a Streamlit web app.

How It Works
Baseline embeddings — Images are passed through a pretrained ResNet50 (ImageNet weights, no top layer) to extract generic 2048-dim visual feature vectors.
Similarity search — Embeddings are indexed with FAISS (IndexFlatIP, cosine similarity via L2-normalized vectors) to retrieve the top-K most visually similar products for a query image.
Triplet generation — Using product category labels (articleType), the pipeline builds (anchor, positive, negative) image triplets: anchor and positive share a category, negative comes from a different category.
Siamese network fine-tuning — A shared-weight embedding model (ResNet50 backbone, partially unfrozen + Dense(256) + L2 normalization) is trained with a custom triplet loss (train_step) so that same-category images are pulled closer together and different-category images are pushed apart in embedding space.
Re-indexing & evaluation — The fine-tuned 256-dim embeddings are re-indexed in FAISS, and retrieval quality is measured with Precision@K and Recall@K.
Deployment — A Streamlit app (app.py) provides an upload-an-image UI, exposed publicly via localtunnel (designed to run in Google Colab).
Tech Stack
Component	Tool
Feature extraction / embedding model	TensorFlow / Keras, ResNet50
Metric learning	Custom Siamese network + Triplet Loss
Similarity search	FAISS (faiss-cpu)
Data	Fashion Product Images (Small) — Kaggle
Web app	Streamlit
Tunneling (Colab deployment)	localtunnel
Data handling	pandas, NumPy
Visualization	Matplotlib
Project Structure (Notebook Sections)
Setup & Data Download — Kaggle API config, dataset download and unzip.
Baseline Feature Extractor — Pretrained ResNet50 embedding function.
Triplet Generator (v1) — Random triplet sampling utility.
Siamese Model Skeleton — Initial multi-input Siamese architecture.
FAISS Baseline Search — Index building and top-K retrieval on raw ResNet50 features.
Retrieval Visualization — Query + recommended images plotted side by side.
Metadata Loading — Load styles.csv product metadata.
Balanced Subset Creation — Sample a balanced dataset across top 5 product categories.
Triplet Generator (v2) — Full triplet dataset generation over the balanced subset.
Triplet Visualization — Sanity-check anchor/positive/negative samples.
Fine-tunable Siamese Network — ResNet50 (partially unfrozen) + Dense(256) + L2-norm embedding head.
Custom Triplet Loss Training Loop — SiameseModel subclass with custom train_step.
Data Pipeline & Training — tf.data pipeline + model training (5 epochs baseline).
Fine-tuned Retrieval & Visualization — Re-index and compare recommendations post fine-tuning.
Evaluation Metrics — Precision@5 / Recall@5 on retrieval results.
Streamlit App — app.py UI for image upload and product recommendation.
Deployment — Run the app publicly via localtunnel (Colab-friendly).
Getting Started

Prerequisites

pip install tensorflow faiss-cpu streamlit kaggle pandas numpy matplotlib pillow

1. Get the Data — Obtain a Kaggle API token (kaggle.json), then:

kaggle datasets download -d paramaggarwal/fashion-product-images-small
unzip -q fashion-product-images-small.zip -d dataset

2. Run the Notebook — Open Final_project_of_celebal.ipynb and run the cells sequentially: cells 1–7 (baseline retrieval demo), 8–15 (build subset, generate triplets, train the fine-tuned model), 16–17 (evaluate), 18–20 (launch the Streamlit app).

3. Launch the App

streamlit run app.py

Note: app.py currently provides the UI scaffold only. To make it fully functional, save the trained embedding_model, fine_tuned_embeddings, and filtered_image_paths to disk, then load them inside app.py to perform live FAISS search.

Evaluation
Precision@K — fraction of top-K retrieved items in the same category as the query.
Recall@K — fraction of all same-category items successfully retrieved in the top-K.
Future Improvements
Persist the trained model/index for a fully functional deployed app.
Hard-negative mining for better triplet training.
Train longer / unfreeze more layers with an LR schedule.
Evaluate on the full dataset, not just a 5-category subset.
Add category/style filters and a proper catalog UI.
Acknowledgements
Dataset: Fashion Product Images (Small) by Param Aggarwal (Kaggle).
Pretrained backbone: ResNet50 (ImageNet weights, Keras Applications).
