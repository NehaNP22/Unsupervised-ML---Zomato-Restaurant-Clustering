### Problem statement (classification pipeline)

#### **Objective**
Build a supervised classification model that predicts whether a *given Zomato review* should be considered **good** or **bad** using only the **engineered numeric features** available in `reviews_df`. The model outputs a probability score `p_good` that a review is good, which can be used for ranking, monitoring, and downstream aggregation to restaurant-level quality signals.

#### **Target definition**
Define a binary target variable:

`GoodReview = 1` if `Rating_Num >= 4`  
`GoodReview = 0` if `Rating_Num < 4`

This turns the star rating into a clear, business-friendly outcome that approximates “positive customer experience.”

#### **Input data and features**
Use the existing `reviews_df` table at the **review level**. Each row represents one review, and the model uses **engineered numeric features** derived from metadata and/or the review text (already precomputed and stored as numeric columns in `reviews_df`). These features are intended to capture signals like reviewer behavior, sentiment strength, review length/structure, timing effects, and other numeric indicators.

The deliverable explicitly excludes raw free-text modeling here (no TF-IDF/embeddings); it’s a baseline using the numeric feature set already engineered.

#### **Model choice**
Train a strong baseline classifier using:

`HistGradientBoostingClassifier`

This model is selected because it is typically high-performing on tabular data, handles nonlinearities and interactions well, and requires minimal feature scaling.

#### **Model output**
For each review, produce:

`p_good = P(GoodReview = 1 | engineered numeric features)`

This probability can be thresholded into a binary decision (good/bad) or used directly as a continuous score.

#### **Success criteria**
A successful baseline should demonstrate meaningful predictive performance on held-out data using standard classification metrics such as ROC-AUC and PR-AUC, and should be interpretable enough to identify which engineered numeric signals are driving predictions (e.g., via permutation importance or SHAP).

#### **Important caveat for a real deliverable**
Because the target is derived from `Rating_Num`, any engineered feature that directly includes or is computed using the rating (for example, reviewer average rating computed using the full dataset, or differences vs the current rating) can introduce **target leakage** and artificially inflate performance. A proper deliverable must ensure all features reflect information that would be available at prediction time and are computed without peeking at the label for the same row (or future data).


### Challenges
**Data leakage and label leakage.** The dataset contained columns derived from the target (e.g., reviewer-average ratings, engineered columns with `Rating` in the name), which would artificially inflate model performance if left in place.  
**Missing / all-NaN columns.** Columns such as `Cost` and `Followers` were entirely empty for many rows and had to be removed to avoid training warnings and unstable behavior.  
**Ranking group mismatch and candidate-set consistency.** LightGBM ranker (`lambdarank`) requires consistent group definitions; reviewers had varying candidate counts which caused group-sum warnings and required filtering/reshaping so train/test groups matched expected sizes.  
**Class imbalance and noisy labels.** Reviews are skewed across ratings; deciding a relevance threshold (we used `>= 4`) and handling imbalance affected metric selection and model calibration.  
**Feature quality & engineering workload.** Useful signals came from a mix of numeric, categorical and text-derived features (sentiment), requiring careful engineering and selection (and pattern-based dropping to avoid leakage).  
**Evaluation complexity for ranking vs. classification.** Evaluating NDCG/hitrate vs. ROC/PR needs separate pipelines and careful splitting strategies (user-level split to avoid information leakage between train and test).  
**Reproducibility & artifacts management.** Multiple models, saved CSVs, and experiment artifacts needed consistent naming and storage for reproducibility.

### Actions
**Leakage removal and pattern-based column dropping.** Explicitly removed known leakage columns (e.g., reviewer aggregates and any column containing rating tokens) and applied pattern-based filtering to numeric columns.  
**Dropped all-NaN columns.** Detected and removed columns that were entirely missing in the training split (e.g., `Cost`, `Followers`) so training runs cleanly.  
**User-level splitting and candidate-set filtering for ranking.** Split data by reviewer (group-aware) and filtered reviewers so candidate set sizes were consistent across train/test to prevent group-sum warnings during LightGBM ranking.  
**Two modeling paths.** 
- Ranking: trained a LightGBM ranker (`lambdarank`) using NDCG@3/4 as evaluation and produced top-1/top-3 recommendations per reviewer from candidate sets.  
- Classification: built a robust review-level `GoodReview` classifier using a pipeline (median imputation + HistGradientBoostingClassifier) to predict probability a review is good.  
**Evaluation & importance analysis.** Computed ranking hit rates / NDCG where applicable; for classification computed ROC AUC, PR AUC, confusion matrix and classification report and used permutation importance to produce a clean importance table.  
**Saved artifacts.** Exported recommendation lists and prediction / importance CSVs for downstream use: `top1_recommendations_per_reviewer.csv`, `top3_recommendations_per_reviewer.csv`, `review_level_p_good_no_leakage_clean.csv`, `permutation_importance_no_leakage_clean.csv`.

### Outcomes
**Ranking outputs created.** Top-1 and Top-3 recommendation files were generated and saved. In the immediate run shown, the hitrate metrics computed on that candidate set run were `hitrate@1 = 0.0` and `hitrate@3 = 0.0` (this indicates either the candidate sets or the relevance threshold produced no relevant items in top picks for that particular evaluation slice). Note: ranking performance can vary a lot with candidate sampling, the relevance threshold, and how reviewers/candidates were filtered.  
**Strong classification performance.** The cleaned review-level classifier achieved:
- `ROC AUC = 0.935`  
- `PR AUC = 0.958`  
- Confusion matrix (test) printed as `[[571 174], [ 96 1159]]` with overall accuracy ≈ `0.865`.  
This shows the model can predict whether a review is “good” (`>=4`) with high discriminative power after removing leakage.  
**Feature importance insights.** Permutation importance highlighted text-sentiment features (e.g., `Vader_Compound`, TB polarity features) among top contributors, confirming the value of textual sentiment signals. The all-NaN columns removed were listed as `['Cost', 'Followers']`.  
**Clean artifacts for reuse.** Saved CSVs make it straightforward to inspect reviewer-level predictions, recommendation lists, and the importance table for productization or further analysis.


