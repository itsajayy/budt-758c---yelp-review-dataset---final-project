# Yelp Review Usefulness Prediction

## Overview

This project predicts whether a Yelp review is likely to be classified as **highly useful**. It was completed for **BUDT758T** as a predictive analytics and machine learning competition project.

Our team placed **second in the class competition** by building a high-performing blended machine learning model that identified useful reviews while keeping the false-positive rate below the required 10% threshold.

The final model achieved:

```text
True Positive Rate: 0.735227
False Positive Rate: 0.096996
```

The model successfully captured approximately **73.5% of highly useful reviews** while staying under the project’s false-positive rate constraint.

## Business Problem

Online review platforms such as Yelp rely on user-generated reviews to help customers make better decisions. However, not all reviews are equally useful. Some reviews provide detailed, credible, and actionable information, while others are vague, biased, or low quality.

The business problem is:

**How can Yelp predict which reviews are likely to be highly useful before enough users have voted on them?**

Solving this problem helps Yelp improve the quality of review rankings, reduce user search effort, and increase trust in the platform.

## Business Value

A real-time review usefulness prediction model can create value for multiple stakeholders.

### For Yelp

* Rank useful reviews higher on business pages
* Improve user trust and review quality
* Reduce reliance on waiting for usefulness votes
* Improve review discovery and user experience
* Support moderation and recommendation systems

### For Yelp Users

* See more informative reviews first
* Spend less time filtering through low-quality reviews
* Make better purchasing and dining decisions

### For Businesses

* Understand what types of reviews influence customers
* Identify highly impactful positive and negative reviews
* Improve customer feedback collection strategies
* Monitor reputation risk from highly useful negative reviews

## Project Objective

The goal was to build a classification model that predicts whether a review is highly useful.

The project’s key evaluation constraint was:

```text
False Positive Rate must be less than or equal to 10%
```

Because incorrectly promoting low-quality reviews can damage user trust, the model was optimized around **TPR at approximately 10% FPR**, rather than accuracy alone.

## Target Variable

The target variable identifies whether a Yelp review belongs to the highly useful class.

Approximately **15% of reviews** were identified as highly useful, making this an imbalanced classification problem.

## Data Sources

The project used Yelp data containing business, user, review, and tip-level information.

An external holiday dataset was also added to create time-based features related to U.S. holidays.

### Main Data Sources

* Yelp business data
* Yelp user data
* Yelp review data
* Yelp tip data
* External U.S. holiday dataset

## Feature Engineering

Feature engineering was one of the most important parts of this project. The final model used more than 130 engineered variables.

Features were created from five major areas:

1. Business characteristics
2. User behavior
3. Review text
4. Review timing
5. External holiday context

## Business Features

Business-level features included:

* Business review count
* Business star rating
* Business category count
* Latitude and longitude
* Days open per week
* Average hours open per day
* Restaurant attributes
* Parking availability
* Delivery and takeout availability
* Outdoor seating
* Wheelchair accessibility
* Price range
* Alcohol availability
* Noise level
* WiFi availability

Categorical features were cleaned and transformed using one-hot encoding.

## User Features

User-level features captured reviewer history and credibility.

Examples included:

* User review count
* User friend count
* User fan count
* User elite count
* Years since joining Yelp
* Compliment count
* Tip count
* Elite status duration

One important finding was that over **92% of users were not elite**, but among elite users, many had held elite status for 3–5 years.

## Review and Text Features

Review-level features captured the content and structure of each review.

Examples included:

* Review text length
* Word count
* Positive word count
* Negative word count
* Star rating
* Whether the review was positive
* Whether the review was negative
* Whether the review used an extreme star rating
* Review day of week
* Weekend indicator

The project found that lower-star reviews had a higher top-useful rate, suggesting that negative reviews may attract more user attention when they provide detailed information.

## Interaction Features

Several interaction features were created to capture relationships between user, business, and review behavior.

Examples included:

* Difference between review stars and business average stars
* Difference between review stars and user average stars
* Absolute star difference from business rating
* Absolute star difference from user rating
* Review length multiplied by user review count
* Text length multiplied by business review count
* Word count multiplied by business review count
* User tip count multiplied by review text length

These features helped the model understand whether a review was unusually positive, unusually negative, detailed, or written by an experienced reviewer.

## Holiday Features

An external U.S. holiday dataset was used to create temporal context features.

Holiday features included:

* Whether the review was written on a U.S. holiday
* Days to nearest holiday
* Days since previous holiday
* Days until next holiday
* Whether the review was within a 3-day holiday window
* Whether the review was within a 7-day holiday window

Holiday features only produced a very small lift in logistic regression, but they were retained in the final pipeline for additional temporal context.

## Exploratory Data Analysis

The project explored several patterns in the Yelp dataset.

Key EDA findings included:

* Lower-star reviews had a higher top-useful rate.
* Food and restaurant businesses dominated the dataset.
* Mexican, pizza, and Chinese restaurants were among the most common categories.
* Business review count had high variability.
* Missing business hours were concentrated in automotive, pets, and salons/spas.
* Pennsylvania had the largest number of businesses.
* User review count showed the largest difference between useful and non-useful reviews.
* Holiday-window features showed limited difference between useful and non-useful reviews.
* Most users were not elite.
* Among elite users, most held elite status for 3–5 years.

## Modeling Approach

The project tested multiple model families, starting with simple baselines and progressing toward advanced ensemble models.

Models tested included:

* Logistic Regression
* Decision Tree
* Random Forest
* LightGBM
* HistGradientBoosting
* XGBoost
* MLP
* Blended ensemble models

The final submitted model was a weighted blend of top-performing models.

## Model Evaluation Metric

The main evaluation metric was:

```text
TPR at FPR ≤ 10%
```

This means the model was judged by how many highly useful reviews it could correctly identify while keeping the false-positive rate below 10%.

This metric was more important than accuracy because the business cost of incorrectly promoting low-quality reviews is high.

## Baseline Models

### Business-Only Logistic Regression

The first baseline used only business-level features.

```text
AUC: 0.5938
FPR: 0.1000
TPR: 0.2020
```

This model performed only slightly better than random ranking, showing that business-level features alone did not contain enough predictive signal.

### Business-Only Decision Tree

A decision tree was trained on the same business-only features.

```text
AUC: 0.7017
FPR: 0.1000
TPR: 0.3128
```

The decision tree improved performance by capturing nonlinear interactions, but the business-only feature set was still not sufficient.

## Full Feature Models

### Random Forest

The Random Forest model used the full merged feature set.

```text
Validation AUC: 0.8685
Default FPR: 0.0436
Default Recall: 0.4267
Precision for class 1: 0.57
F1-score for class 1: 0.49
```

The model performed much better than the business-only baselines, but the learning curve showed overfitting because training AUC was nearly perfect while validation AUC was lower.

### LightGBM Cross-Validation

LightGBM was introduced because it performs well on high-dimensional tabular classification problems.

Earlier LightGBM CV result:

```text
Mean AUC: 0.9069
Mean FPR: 0.0969
Mean TPR: 0.6852
```

This was the strongest single-model result at that stage.

### Improved LightGBM with 5-Fold CV

An improved LightGBM model used stronger features and 5-fold stratified cross-validation with early stopping.

```text
Mean AUC: 0.9109
Mean TPR at 10% FPR: 0.7054
Standard deviation of TPR: 0.0049
```

Fold-level results were stable, with validation AUC values around 0.909–0.913 and TPR values around 0.700–0.714.

This became the strongest standalone model in the report.

### HistGradientBoosting

HistGradientBoosting was also tested.

```text
Mean AUC: 0.8953
OOF AUC: 0.8953
Mean FPR: 0.0950
Mean TPR: 0.6469
```

It was competitive but did not outperform LightGBM.

### Full-Feature Logistic Regression

A logistic regression model using the full engineered feature set performed surprisingly well.

Without holiday features:

```text
AUC: 0.880496
FPR: 0.099969
TPR: 0.637448
Accuracy: 0.868210
```

With holiday features:

```text
AUC: 0.880487
FPR: 0.099997
TPR: 0.637500
Accuracy: 0.868191
```

The holiday features produced only a negligible improvement, but the model confirmed that feature engineering dramatically improved the linear baseline.

## Final Blended Model

The final model was a **Top-K blended ensemble** combining top-performing models from the model zoo.

The blend included:

* LightGBM variants
* XGBoost variants
* MLP ensemble models

Candidate blend weights were optimized through random search.

The process included:

```text
30,000 candidate blends on a sampled validation set
Top 300 candidates evaluated on full validation set
```

### Final Blend Performance

```text
Final TPR: 0.735227
Final FPR: 0.096996
Reported validation AUC: 0.879
OOF AUC: 0.8953
```

This was the best final operating-point performance and satisfied the false-positive rate constraint.

## Competition Result

Our team placed **second in the class competition**.

The final model achieved a strong operating-point result by identifying approximately **73.5% of highly useful reviews** while keeping false positives below **10%**.

This result was competitive because the model was optimized for the project’s actual business constraint rather than only maximizing AUC.

## Key Findings

* Business-only features were not enough to predict review usefulness well.
* User behavior and review text features created the largest performance improvement.
* Lower-star reviews were more likely to become highly useful.
* Review length and user review history were important signals.
* Differences between review rating, business rating, and user rating helped capture review distinctiveness.
* Holiday features provided limited lift but added temporal context.
* LightGBM was the strongest standalone model.
* Blending LightGBM, XGBoost, and MLP models produced the best final TPR at the required FPR threshold.

## Tools & Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* MLP / Neural Network models
* Logistic Regression
* Random Forest
* HistGradientBoosting
* Feature engineering
* Text engineering
* Target encoding
* Cross-validation
* Model blending
* Kaggle
* VSCode

## Limitations

The project had several limitations:

* Models were computationally expensive to run.
* Local machines and Google Colab were not sufficient for some experiments.
* The team eventually used Kaggle to handle larger subsets of data.
* Some feature engineering work happened separately across team members, which made model comparison harder.
* Holiday features had minimal impact.
* The blended model improved the target metric but was less interpretable than simpler models.
* Runtime constraints limited deeper hyperparameter tuning and ensemble experimentation.

## Future Improvements

Future improvements could include:

* More coordinated shared feature engineering
* Additional text embeddings or transformer-based review representations
* More systematic hyperparameter tuning
* More advanced stacking instead of weighted blending
* Better calibration of predicted probabilities
* SHAP-based explainability for business interpretation
* Deployment as a real-time usefulness scoring API
* Review ranking simulation to measure product impact
* Separate modeling for positive and negative reviews
* Industry-specific usefulness models by business category

## Reflection

The project demonstrated the importance of feature engineering, collaboration, and model evaluation under business constraints. The largest performance gains came from merging business, user, review, and tip-level features rather than relying on business data alone.

The team learned that computational planning matters for large-scale machine learning projects. Early experiments in Google Colab were limited by runtime and memory constraints, so the team moved to local VSCode environments and later Kaggle to run larger models.

If starting again, the team would define a shared feature engineering pipeline earlier and collaborate more closely on cleaned datasets, feature definitions, and model outputs.

## Conclusion

This project built a high-performing machine learning model to predict highly useful Yelp reviews. The final blended model combined LightGBM, XGBoost, and MLP models and achieved:

```text
TPR = 0.735227
FPR = 0.096996
```

By staying under the 10% false-positive rate constraint while identifying over 73% of highly useful reviews, the model provides a strong foundation for improving Yelp review ranking, user trust, and content discovery.

Our team placed **second in the class competition**, making this one of the strongest projects in the course.
