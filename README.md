# DSN Bootcamp Qualification Hackathon 2026 — ML Track

### Predicting `total_sales` with Leakage-Safe Feature Engineering and CatBoost

> **Competition:** DSN Bootcamp Qualification Hackathon 2026 — ML Track
> **Author:** Adenike Adewumi
> **Date:** September 2026
> **Task:** Regression
> **Evaluation Metric:** RMSE

---

## 📌 Overview

This repository contains my complete machine learning pipeline for the **DSN Bootcamp Qualification Hackathon 2026 — ML Track**, where the objective was to predict `total_sales` for products across different stores.

Rather than stopping at a baseline model, I treated the competition as an iterative machine learning experiment. The notebook documents the progression from exploratory data analysis and an initial one-hot encoded baseline through feature engineering, leakage-safe target encoding, model comparison, hyperparameter tuning, and finally seed averaging.

One of the main goals of this project was not simply to find a model that performed well, but to understand **why** particular decisions improved or hurt performance.

The final pipeline uses a **tuned CatBoost regression model**, trained using the full training data and averaged across **15 random seeds**.

---

## 🎯 Problem Statement

The dataset contains product- and store-level information, including:

* Product characteristics
* Product category
* Product price
* Store characteristics
* Store location information
* Store format
* Product and store identifiers

The target variable is:

```text
total_sales
```

The task is therefore a supervised regression problem to predict the total sales

The competition evaluates predictions using **Root Mean Squared Error (RMSE)**:

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

Lower RMSE indicates better predictive performance.

---

## 📊 Dataset

The competition data contains:

| Dataset  |  Rows | Columns |
| -------- | ----: | ------: |
| Training | 6,818 |      13 |
| Test     | 1,705 |      12 |

The training data contains the target column `total_sales`, while the test data does not.

### Main Features

| Feature               | Description / Role                  |
| --------------------- | ----------------------------------- |
| `product_code`        | High-cardinality product identifier |
| `product_weight_kg`   | Product weight                      |
| `fat_content`         | Product fat-content category        |
| `shelf_visibility`    | Product shelf visibility            |
| `product_category`    | Product category                    |
| `product_price`       | Product price                       |
| `store_code`          | Store identifier                    |
| `store_age_years`     | Age of the store                    |
| `store_size`          | Store size category                 |
| `store_location_tier` | Store location tier                 |
| `store_format`        | Store format                        |
| `total_sales`         | Target variable                     |

---

# 🔬 Approach

The project follows the workflow below:

```text
Competition Data
       │
       ▼
Data Loading
       │
       ▼
Exploratory Data Analysis
       │
       ├── Missing-value analysis
       ├── Target distribution
       ├── Category inspection
       ├── Correlation analysis
       └── Outlier / relationship analysis
       │
       ▼
Data Cleaning
       │
       ├── Category normalization
       ├── Missing-value treatment
       └── Invalid zero-value treatment
       │
       ▼
Feature Engineering
       │
       ├── Price × Store Age
       ├── Price vs Category Mean
       ├── Price vs Store Format Mean
       ├── Price per Kg
       └── Store Average Price
       │
       ▼
Leakage-Safe Target Encoding
       │
       ▼
5-Fold Cross-Validation
       │
       ├── CatBoost
       ├── LightGBM
       ├── XGBoost
       └── Gradient Boosting
       │
       ▼
Loss-Function Experiments
       │
       ▼
Feature Ablation
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Tuned CatBoost
       │
       ▼
15-Seed Prediction Averaging
       │
       ▼
Final Submission
```

---

# 🔎 Exploratory Data Analysis

A substantial part of the project was spent understanding the data before choosing a modeling strategy.

Some of the most important findings were:

### 1. Inconsistent product-category labels

`product_category` contained categories written in different cases, such as:

```text
Snack Foods
snack foods
SNACK FOODS
```

Although these appear to represent the same category, treating them as separate categories artificially increases the number of groups.

I therefore normalized the category values before performing downstream group-based operations.

---

### 2. Missing values

Two important columns contained substantial missingness:

* `store_size`
* `product_weight_kg`

Instead of applying a blanket global median everywhere, the preprocessing strategy used the structure of the data to make the imputations more meaningful.

---

### 3. Zero shelf visibility

`shelf_visibility` contained a noticeable concentration of exactly zero values.

Since a product physically present on a shelf having literally zero visibility is questionable, these values were treated as missing rather than automatically assuming that zero represented a genuine measurement.

The missing values were then imputed using category-level information.

---

### 4. High-cardinality `product_code`

`product_code` contained approximately **1,555 unique values**.

One-hot encoding this feature caused a substantial increase in dimensionality and contributed to overfitting in an early experiment.

Rather than creating hundreds of sparse indicator variables, I used **smoothed, leakage-safe target encoding**.

---

### 5. Store format showed a strong relationship with sales

EDA showed a pronounced relationship between `store_format` and the target.

Interestingly, some categorical variables that might intuitively appear ordinal were **not actually monotonic** in their relationship with sales.

For example, I initially considered treating `store_size` and `store_location_tier` as ordered variables. Examination of the actual target distributions showed that this assumption was not justified, so I avoided forcing an artificial ordinal relationship onto them.

---

# 🧠 Feature Engineering

Several engineered features were created based directly on observations from the EDA.

### `price_x_store_age`

Captures the interaction between product price and store age.

### `price_vs_category_mean`

Measures how a product's price compares with the typical price within its product category.

### `price_vs_store_format_mean`

Measures the product price relative to the typical price associated with its store format.

### `price_per_kg`

Provides a normalized measure of product price relative to weight.

### `store_avg_price`

Captures the average pricing level associated with the store.

The purpose of these features was not to generate a large number of arbitrary variables, but to encode relationships that appeared meaningful during exploration.

---

# 🔐 Leakage-Safe Target Encoding

One of the central parts of the final pipeline is **target encoding**.

For a categorical variable, target encoding replaces a category with information derived from the target:

$$
category \rightarrow E[y \mid category]
$$

However, calculating these statistics directly on the entire training set can cause **target leakage**, because the target value of the validation observation can indirectly influence its own encoded feature.

To avoid this, the notebook performs target encoding within the cross-validation structure.

The encoding is also **smoothed toward the global target mean**, particularly for categories with relatively few observations.

The smoothing strategy was stronger for high-cardinality `product_code` because each product code had relatively few observations.

---

# 🤖 Models Tested

Several tree-based regression approaches were evaluated using cross-validation, including:

* **CatBoost**
* **LightGBM**
* **XGBoost**
* **Gradient Boosting**

Rather than relying on a single train/validation split, model comparisons were performed using **5-fold cross-validation**.

This provided a more reliable estimate of how each approach generalized across different subsets of the training data.

---

# 🧪 Experiments

A major focus of this project was testing assumptions instead of simply keeping techniques because they are commonly recommended.

## Log-transforming the target

Because `total_sales` was right-skewed, I tested:

```python
log1p(total_sales)
```

and trained models on the transformed target.

However, cross-validation showed that the log-transformed approach performed worse than training directly on the raw target.

The raw target was therefore retained.

---

## One-hot encoding vs target encoding

An early baseline used one-hot encoding.

This created a large feature space, particularly because of `product_code`.

The experiment showed that target encoding provided a more useful representation for the categorical variables while avoiding the dimensionality explosion of one-hot encoding.

---

## Huber loss vs RMSE

I also tested robust Huber-style objectives against standard squared-error/RMSE objectives.

Huber loss produced a slightly better cross-validation result in one experiment, but the improvement on the actual leaderboard was small enough that I treated the difference as practically close.

The final pipeline therefore used the simpler RMSE-oriented setup.

---

## Feature ablation

Several features were individually removed to test whether they were contributing useful information.

An interesting result was that some features appeared removable when tested individually, but removing several of them together actually **hurt cross-validation performance**.

This suggested that their information was partially overlapping but still complementary.

As a result, I retained the full feature set in the final pipeline.

---

## Entity embeddings

I experimented with using neural-network-based entity embeddings for `product_code` as an alternative to target encoding.

This approach performed worse in cross-validation.

The likely explanation is that each product code has relatively few observations, making it difficult for a neural network to learn stable embeddings without overfitting.

The embedding implementation is retained in the notebook as a documented experiment, but it is **not used in the final submission pipeline**.

---

## Model blending

I also tested blending predictions from multiple model types.

Both:

* flat averaging, and
* CV-weighted averaging

performed worse than the strongest individual model on the actual leaderboard.

I therefore abandoned multi-model blending.

---

# 🌱 Seed Averaging

One technique that did provide a useful improvement was averaging predictions from multiple random seeds of the **same tuned model**.

The final pipeline trains the tuned CatBoost model using:

```text
15 random seeds
```

and averages the resulting predictions:

$$
\hat{y}_{final} = \frac{1}{15} \sum_{i=1}^{15} \hat{y}_i
$$

This is different from blending different model architectures because every prediction comes from the same tuned model configuration.

The notebook observed diminishing returns beyond approximately 10–15 seeds, so the final pipeline uses 15.

---

# 🏆 Performance Progression

One of the most useful parts of this project was keeping track of what actually improved performance.

| Experiment                                    | Public LB RMSE | Notes                                                 |
| --------------------------------------------- | -------------: | ----------------------------------------------------- |
| Initial baseline                              |    **1094.09** | Plain train/validation split + one-hot encoding       |
| Early flat model blend                        |    **1127.65** | Worse than the individual model                       |
| Category cleanup + target encoding + CatBoost |    **1078.77** | Major improvement                                     |
| CV-weighted blend                             |    **1088.77** | Still worse than the individual model                 |
| Further model tuning                          |    **1073.60** | Additional improvement                                |
| Seed averaging                                |    **1072.23** | Best score before the consolidated notebook           |
| Consolidated notebook                         |    **1072.91** | Full feature set + tuned CatBoost + 15-seed averaging |

> **Note:** The scores above are the competition/leaderboard results recorded during the project and are included to document the experimentation process rather than to represent a guaranteed reproducible leaderboard result.

The key lesson from these experiments was that **better data understanding and validation mattered more than simply adding more models**.

---

# 🏁 Final Model

The final pipeline uses:

### Model

**CatBoost Regressor**

### Training

* Full training dataset
* Tuned hyperparameters
* Native categorical handling
* 15 different random seeds

### Prediction

Predictions from the 15 models are averaged to produce the final test prediction.

Negative predictions are clipped to zero before averaging.

The final predictions are then written to:

```text
submission_final.csv
```

with the competition's required ID and prediction columns.

---

# 📁 Repository Structure

A simple repository structure for this project is:

```text
.
├── README.md
└──  dsn_hackathon_ml_track_adenike_adewumi.ipynb
```

The main notebook contains the complete workflow, including:

* Data loading
* Exploratory analysis
* Data cleaning
* Feature engineering
* Target encoding
* Cross-validation
* Model comparison
* Loss-function experiments
* Feature ablation
* Hyperparameter tuning
* Seed averaging
* Feature importance analysis
* Final prediction generation

---


# 🛠️ Technologies Used

| Technology   | Purpose                                     |
| ------------ | ------------------------------------------- |
| Python       | Main programming language                   |
| Pandas       | Data manipulation                           |
| NumPy        | Numerical computation                       |
| Scikit-learn | Cross-validation, preprocessing and metrics |
| CatBoost     | Final regression model                      |
| LightGBM     | Model comparison                            |
| XGBoost      | Model comparison                            |
| Matplotlib   | Visualization                               |
| Seaborn      | Exploratory visualization                   |
| KaggleHub    | Competition data access                     |
| Google Colab | Development environment                     |

---

# 💡 Key Lessons

This competition reinforced several practical machine learning lessons for me:

### 1. EDA is part of modeling

Some of the biggest improvements did not come from changing the algorithm. They came from noticing problems in the data, particularly inconsistent categorical labels and the behavior of individual store formats.

### 2. Don't assume a categorical variable is ordinal

A variable containing values such as `Small`, `Medium`, and `Large` may look naturally ordered, but the relationship with the target needs to be checked before imposing that structure.

### 3. High-cardinality categorical variables require care

One-hot encoding is not always the best choice. For `product_code`, target encoding provided a more compact representation.

### 4. Leakage prevention matters

Target encoding can be powerful, but it can also leak target information if implemented incorrectly. Encoding statistics need to be calculated using only the appropriate training portion of each fold.

### 5. Cross-validation is more useful than intuition

Several ideas that sounded reasonable performed worse when actually tested:

* Target log transformation
* Entity embeddings
* Multi-model blending
* Removing apparently redundant features

Testing them was more informative than assuming they would work.

### 6. A simpler final pipeline can beat a more complicated one

The final solution did not require a neural network or a complicated ensemble of unrelated models.

A carefully tuned CatBoost model, supported by good preprocessing and feature engineering, was sufficient.

---

