# Repository for learning Data Science and Machine Learning

You can find various resources with specific names in the folders.

Here is the full roadmap that can be followed to learn ML. ( I will not be following the flow of this map, but will be trying to complete the topics.)

# In-depth Study Plan for Learning Machine Learning (ML)

This plan is designed for a Computer Science graduate who wants to master ML from end to end, including creating models, deploying, and maintaining them.

## Phase 1: Foundational Knowledge (4–6 weeks)

**Goal:** Establish solid mathematical and programming fundamentals.  
### Topics:
1. **Mathematics for ML**  
   - **Linear Algebra**  
     - Vectors, Matrices, Operations (Dot Product, Cross Product), Eigenvalues, Eigenvectors
     - Matrix Factorization (SVD, PCA)
   - **Probability and Statistics**  
     - Probability Distributions (Normal, Binomial, Poisson)
     - Bayes’ Theorem, Conditional Probability, Likelihood
     - Hypothesis Testing, Confidence Intervals
     - Descriptive vs. Inferential Statistics
   - **Calculus**  
     - Differentiation: Gradients, Chain Rule, Partial Derivatives
     - Optimization techniques (Gradient Descent, Stochastic Gradient Descent)
   - **Optimization**  
     - Loss Functions (MSE, Cross-Entropy, Hinge Loss)
     - Lagrange Multipliers, Convex Optimization

2. **Programming in Python**  
   - **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
   - **Object-Oriented Programming** for cleaner code.
   - **Version Control**: Git and GitHub for managing code and collaborative work.
   - **Jupyter Notebooks**: Essential for writing and documenting experiments.

---

## Phase 2: Core Machine Learning Concepts (8–10 weeks)

**Goal:** Learn core algorithms and techniques in machine learning.  
### Topics:
1. **Supervised Learning**
   - **Regression**
     - Linear Regression, Polynomial Regression
     - Regularization (Ridge, Lasso)
     - Metrics (R², MSE, MAE)
   - **Classification**  
     - Logistic Regression, Decision Trees, Random Forests, k-NN
     - Support Vector Machines (SVM)
     - Evaluation metrics (Precision, Recall, F1-Score, ROC, AUC)
  
2. **Unsupervised Learning**  
   - **Clustering**  
     - k-Means, Hierarchical Clustering, DBSCAN
   - **Dimensionality Reduction**  
     - Principal Component Analysis (PCA)
     - t-SNE, UMAP
   - **Anomaly Detection**

3. **Ensemble Methods**  
   - Bagging (e.g., Random Forest)
   - Boosting (e.g., AdaBoost, XGBoost, LightGBM, CatBoost)

4. **Model Evaluation and Tuning**
   - **Cross-validation** (k-fold, Stratified)
   - **Hyperparameter Tuning**  
     - Grid Search, Random Search
     - Bayesian Optimization (e.g., Optuna, Hyperopt)

5. **Feature Engineering**  
   - Handling Missing Data, Outliers
   - Encoding Categorical Variables (One-Hot, Label Encoding)
   - Feature Scaling (Min-Max, StandardScaler)

### Technologies:  
- **Python**: NumPy, Pandas, Scikit-learn  
- **Model Tuning**: Hyperopt, Optuna  
- **Data Visualization**: Matplotlib, Seaborn

---

## Phase 3: Deep Learning (8–12 weeks)

**Goal:** Learn advanced machine learning concepts using neural networks and deep learning frameworks.  
### Topics:
1. **Neural Networks**  
   - **Basics**: Perceptron, Multilayer Perceptron (MLP)
   - **Activation Functions**: Sigmoid, ReLU, Tanh, Softmax
   - **Training**: Backpropagation, Weight Initialization, Dropout, Batch Normalization
   - **Optimizers**: Adam, RMSProp, SGD

2. **Deep Learning Architectures**  
   - **Convolutional Neural Networks (CNNs)**: Image Processing, Pooling, Padding
   - **Recurrent Neural Networks (RNNs)**: Sequence Data, LSTMs, GRUs  
   - **Transformers**: Attention Mechanism, Self-Attention, Positional Encoding  
     - **BERT**, **GPT** series, **T5**, **Mistral 7B**
   - **Autoencoders**: For Dimensionality Reduction, Denoising
   - **GANs (Generative Adversarial Networks)**: Image Generation, Style Transfer

3. **Transfer Learning**  
   - Using pre-trained models like ResNet, VGG, BERT for fine-tuning.

4. **NLP (Natural Language Processing)**  
   - **Text Preprocessing**: Tokenization, Stemming, Lemmatization, Stopwords
   - **Word Embeddings**: Word2Vec, GloVe, FastText, Transformers
   - **NLP Models**: Seq2Seq, Transformer Models (BERT, GPT), Attention Mechanism

### Technologies:  
- **Deep Learning Frameworks**: TensorFlow, PyTorch  
- **NLP Frameworks**: Hugging Face Transformers  
- **CNN/RNN Frameworks**: Keras, PyTorch  
- **Libraries**: SpaCy, NLTK

---

## Phase 4: Advanced Topics (8–12 weeks)

**Goal:** Explore advanced techniques and emerging trends in ML.  
### Topics:
1. **Reinforcement Learning (RL)**  
   - **Key Concepts**: Markov Decision Process (MDP), Q-learning, Policy Gradient
   - **RL Libraries**: OpenAI Gym, Stable Baselines

2. **Bayesian Methods**  
   - Bayesian Networks, Gaussian Processes

3. **Time Series Analysis**  
   - ARIMA, SARIMA, LSTMs for Time Series
   - Facebook Prophet, Holt-Winters Method
   - Anomaly Detection in Time Series Data

4. **Recommendation Systems**  
   - Collaborative Filtering, Matrix Factorization
   - Content-Based and Hybrid Systems

5. **Graph Neural Networks (GNNs)**  
   - Node Classification, Graph Embedding, Message Passing
  
6. **Model Interpretability (Explainable AI)**  
   - **SHAP**, **LIME** for explaining model predictions

---

## Phase 5: Model Deployment and MLOps (8–10 weeks)

**Goal:** Learn to deploy and maintain ML models in production.  
### Topics:
1. **Model Deployment**
   - **APIs**: Flask, FastAPI  
   - **Containerization**: Docker, Docker Compose  
   - **Model Serialization**: Pickle, Joblib, ONNX  
   - **Cloud Services**: AWS (SageMaker), GCP (Vertex AI), Azure ML
   - **Serverless**: AWS Lambda, Google Cloud Functions

2. **MLOps**  
   - **CI/CD Pipelines**: Jenkins, GitLab CI/CD  
   - **Monitoring & Logging**: Prometheus, Grafana, ELK Stack  
   - **Model Versioning**: DVC, MLflow  
   - **Data Pipelines**: Airflow, Luigi, Prefect  
   - **Feature Stores**: Tecton, Feast

3. **Scaling Models**  
   - **Distributed Training**: Horovod, Distributed TensorFlow  
   - **Hyperparameter Optimization at Scale**: Ray, Hyperopt

### Technologies:  
- **MLOps Tools**: MLflow, DVC  
- **Model Serving**: TensorFlow Serving, TorchServe  
- **Cloud Platforms**: AWS, GCP, Azure  
- **Containerization**: Docker  

---

## Phase 6: Specialization (Optional)

**Goal:** Delve deeper into specific domains like Computer Vision, NLP, or Edge AI.  
### Topics:
1. **Computer Vision**  
   - Object Detection (YOLO, Faster R-CNN)
   - Semantic Segmentation (UNet, Mask R-CNN)

2. **NLP Applications**  
   - Question Answering, Summarization, Translation  
   - Generative AI with transformers like **GPT-4** and **T5**

3. **Edge AI**  
   - Optimizing models for deployment on edge devices (Raspberry Pi, Mobile)

### Technologies:  
- **Edge ML Tools**: TensorFlow Lite, ONNX Runtime  

---

## Summary of Tools & Technologies:

1. **Python Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Keras, Matplotlib, Seaborn
2. **NLP Frameworks**: Hugging Face Transformers, SpaCy, NLTK
3. **Cloud Platforms**: AWS SageMaker, GCP Vertex AI, Azure ML
4. **MLOps Tools**: MLflow, DVC, Kubernetes, Docker, Jenkins, Airflow
5. **APIs & Deployment**: Flask, FastAPI, Docker, TensorFlow Serving, TorchServe

---

## Expected Outcomes:
- **6–12 months**: Full proficiency in ML, from modeling to production deployment.
