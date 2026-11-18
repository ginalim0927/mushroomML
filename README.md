# Mushroom Classification Using Machine Learning

A supervised machine learning project exploring whether a mushroom's physical characteristics can be used to classify it as edible or poisonous. The analysis compares Decision Tree and Random Forest classifiers, examines feature importance, and visualizes relationships between encoded mushroom attributes and the target label.

# Project Overview
The project analyzes 8,124 mushroom records with 23 categorical columns describing mushroom characteristics and classification information. It explores how observable features such as odor, gill characteristics, and spore-print color relate to the edible/poisonous classification.

# Goals
- Explore the distribution and relationships of categorical mushroom attributes.
- Train and compare Decision Tree and Random Forest classification models.
- Identify features that contribute most to model predictions.
- Visualize associations between encoded features and mushroom class using a correlation heatmap.

# Dataset
- Records: 8,124 mushrooms
- Data: Categorical mushroom characteristics and an edible/poisonous class label
- Examples of attributes: Odor, spore-print color, gill size, gill color, and cap characteristics
- Target: Mushroom class (edible or poisonous)
Dataset source: Add the link to the exact dataset download or hosting page used for this project. Dataset versions and column definitions can vary.

# Methodology
1. ## Exploratory Data Analysis
Examined categorical attributes and created a correlation heatmap from numerically encoded features to explore their associations with the target class.
2. ## Classification Models
Trained and evaluated two supervised learning algorithms:
- Decision Tree Classifier — learns a series of feature-based decision rules.
- Random Forest Classifier — combines predictions from multiple decision trees.
3. ## Feature Importance
Compared the models' feature-importance results to identify the attributes that were most influential in classifying mushrooms.
Results
Model	Reported accuracy
Decision Tree	100%
Random Forest	100%


Both models achieved 100% accuracy in the project's evaluation. This suggests the classes were highly separable in the evaluated data, but it does not establish perfect accuracy on new or unseen mushroom species.

# Key Findings
1. Odor emerged as the strongest predictive feature in the model's importance results.
2. Spore-print color, gill size, and gill color were also among the most influential attributes.
3. Several cap-related attributes showed weaker relationships with the class label in the encoded-feature correlation heatmap.
4. The results highlight how feature selection and model interpretability can help explain classification outcomes.
Interpreting the heatmap: Correlations computed after assigning numeric codes to categories can depend on the encoding. They should not be interpreted as definitive evidence that a feature is inherently reliable or unreliable. Model feature importance provides a complementary view, but it also has limitations.

# Visualizations
The project includes a correlation heatmap showing associations between encoded mushroom attributes and the edible/poisonous class label.

Optional: Add an exported image to the repository and embed it here, for example:
![Correlation heatmap](images/correlation-heatmap.png)

# How to Explore the Project
1. Clone or download this repository.
2. Obtain the mushroom dataset used for the analysis and place it where the project code expects it.
3. Open the project's Python script or notebook.
4. Install the dependencies listed in the project's code or environment file, then run the analysis to reproduce the model comparison and visualizations.
The exact script/notebook filename, package versions, and dataset path should be added once the project files are uploaded.
Limitations and Future Improvements
- Validate generalization: Confirm performance on a held-out test set or with cross-validation, and check that preprocessing does not leak information from the test data.
- Review additional metrics: Compare precision, recall, confusion matrices, and per-class performance—not accuracy alone.
- Interpret categorical features carefully: Compare alternative encodings or association measures before drawing conclusions from a correlation heatmap.
- Assess real-world applicability: A perfect score on this dataset should not be used to decide whether a wild mushroom is safe to eat.