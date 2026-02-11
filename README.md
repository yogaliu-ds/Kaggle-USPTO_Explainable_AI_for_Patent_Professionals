# Kaggle-USPTO Explainable AI for Patent Professionals

# Task
## Objective

The primary goal of this competition is to generate Boolean search queries that accurately represent a specific set of patent documents. For a given set of related patents, the model must output a Boolean expression that retrieves the exact same collection of documents from a search engine.

* **Known Targets**: The 50 target patent IDs are provided in the dataset.
* **Mission**: "Reverse-engineer" a Boolean query that forces the search engine to return these 50 specific patents at the top of the results.
* **Test Data Composition**: The `test.csv` file provides a "Seed Patent" and its corresponding **Target Patent Set** (columns `target_1` to `target_50`).
* **Semi-supervised Approach**: Since the target patents are provided even in the test set, this functions as a form of optimization task rather than traditional unsupervised learning. Simulated Annealing uses these 50 provided patents to evaluate and refine the keyword set until it stabilizes.

## Evaluation
* **Metric**: Mean Average Precision at 50 (**mAP@50**).
* The metric measures the relevance and the **ranking order** of the patents retrieved by your query compared to the ground truth set.

---

# Solution

The workflow is divided into three main phases:

### 1. TF-IDF Feature Extraction
Perform TF-IDF analysis specifically on the seed patent and its 50 related target patents, rather than the entire global corpus.

* **What is TF-IDF?**: A statistical measure used to evaluate how important a word is to a document in a collection. Words with high frequency in a specific document but low frequency across the corpus receive higher weights.
* **Two Vectorizers are used**:
    * **Patent Title**: Captures concise, high-level descriptive text.
    * **CPC (Cooperative Patent Classification)**: Captures structured technical metadata representing the patent's specific domain.
* **Vector Representation**: Each patent is converted into a **TF-IDF Vector**—a numerical sequence of individual TF-IDF scores for every word in the vocabulary.

### 2. Query Generation Pipeline
* **Top-k Candidate Selection**: Select the highest-scoring terms based on TF-IDF weights.
    * **Title**: Top 5 words.
    * **CPC**: Top 20 codes.
* **Initial Query Construction**:
    * Join these candidates using the `"OR"` operator.
    * *Example*: `(ti:battery OR ti:electric OR cpc:B60L...)`.
* **Output**: A pool of 25 keyword candidates.

### 3. Simulated Annealing Optimization
**Steps**
1.  **Random Initialization**: Start with a random subset of the top-k keywords.
2.  **Retrieval**: Execute the query via the **Whoosh** search engine to get predicted patents.
3.  **Scoring**: Compare the predicted patents against the **True Labels** (Target Patents) to calculate the **AP@50**.
4.  **Energy Assignment**: Use the negative AP@50 score ($-E$) as the "Energy" for the annealing process.

**The Annealing Mechanism**
1.  **Delta Calculation**: Calculate $\Delta E = E_{new\_set} - E_{old\_set}$ to determine the performance gap between keyword combinations.
2.  **Metropolis Criterion**: Decide whether to accept the new keyword set based on a probability calculation.

> **Key Point**: There is no "Training" in the traditional sense. Instead, we perform **Combinatorial Optimization** on each test sample individually to find the best set of keywords. The Training/Validation sets are primarily used to tune **Hyperparameters** (e.g., $k$ values, temperature range).

**Metropolis Acceptance Probability**
When a keyword is randomly toggled, the change in score results in an energy difference ($\Delta E$):
* If the new combination is better (Score increases, Energy decreases), it is **100% accepted**.
* If the new combination is worse (Score decreases, Energy increases), it is accepted with a **Probability $P$**:
    $$P = e^{-\frac{\Delta E}{T}}$$
* The **Temperature ($T$)** acts as the denominator; as $T$ decreases, the algorithm becomes less tolerant of "bad" moves.

---

### Implementation: State and Updates
* **State**: A binary configuration of the 25 candidate words (included vs. excluded).
* **Neighboring State**: Formed by randomly "toggling" (adding/removing) one keyword.
* **Evaluation**: The Whoosh search engine retrieves documents in real-time to provide the AP@50 feedback.
* **Cooling Schedule**: High initial temperatures allow for broad "exploration" (accepting worse scores), while lower temperatures ensure "exploitation" (converging on the best solution).

### Core Packages
* **Whoosh**: Text indexing and high-speed search engine.
* **Polars**: High-performance data processing.
* **Scikit-learn**: TF-IDF vectorization.
* **Simulated Annealing**: Custom implementation for combinatorial optimization.

---

## Future Directions
1.  Incorporate advanced Boolean operators (AND, NOT, XOR).
2.  Implement more sophisticated feature selection beyond basic TF-IDF.
3.  Optimize the cooling schedule and step count to maximize score within the 60-minute time limit.
4.  Expand support for additional metadata fields such as Abstract or Claims.