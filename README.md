# Sports Prediction Using Machine Learning

This project explores the use of machine learning in sports data analysis. I worked with two classification tasks: predicting cricket match winners and classifying football players based on a simple stamina-based injury-risk label. The main goal was to compare different machine learning algorithms and understand their performance on sports datasets.

## Project Overview

The notebook contains two parts:

1. **Cricket match winner prediction:** Predicts the winning team using match information such as the teams, runs, wickets, and venue.
2. **Football player classification:** Uses player attributes from the FIFA 22 dataset to classify players according to a label created from their stamina value.

Both parts compare **Random Forest, Multinomial Naive Bayes, and XGBoost**.

## Datasets

### Cricket dataset

The included `large_cricket_data.csv` file contains **1,000 matches** and the following columns:

- Match ID
- Team 1 and Team 2
- Team 1 Runs and Team 2 Runs
- Team 1 Wickets and Team 2 Wickets
- Venue
- Winner (target)

### Football dataset

The football experiment uses `players_22.csv`, which is **not included** in this repository's uploaded files. The notebook uses age, stamina, agility, physical attributes, potential, height, and weight.

**Important:** The `injury` label is defined in the notebook as `power_stamina < 50`. It is not based on real injury records, so this part should be understood as a stamina-threshold classification experiment, not a real injury prediction system.

## Methodology

For the cricket experiment, I used label encoding for categorical columns, removed the match ID and target from the input features, and split the data into **80% training and 20% testing**. I then trained three models and compared their accuracy and classification reports.

For the football experiment, I selected player attributes, filled missing feature values using median imputation, and used the same 80/20 train-test split. I also generated a confusion matrix for Random Forest and displayed the top players by potential and agility.

## Results

### Cricket match winner prediction

| Model | Test Accuracy |
|---|---:|
| Random Forest | 79.50% |
| Multinomial Naive Bayes | 16.50% |
| XGBoost | **89.00%** |

XGBoost achieved the highest accuracy in the cricket experiment.

### Football stamina-based classification

| Model | Test Accuracy |
|---|---:|
| Random Forest | 100.00% |
| Multinomial Naive Bayes | 96.31% |
| XGBoost | 100.00% |

These football results should be interpreted carefully because the target is directly calculated from `power_stamina`, which is also included as an input feature. The high scores therefore do not demonstrate an ability to predict real injuries.

**Note:** The cricket accuracy heatmap in the notebook uses manually entered values that differ from the printed model evaluation results. The table above uses the actual printed results.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- Google Colab / Jupyter Notebook

## How to Run

1. Download or clone this repository.
2. Install the required libraries:

   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
   ```

3. Open `Project (1).ipynb` in Jupyter Notebook or Google Colab.
4. Upload `large_cricket_data.csv` and update the cricket CSV path in the notebook if needed.
5. To run the football section, obtain `players_22.csv` separately and update its path as well.
6. Run the cells in order.

The notebook currently uses Google Colab paths such as `/content/large_cricket_data.csv` and `/content/players_22.csv`, so those may need to be changed for local execution.

## Limitations

- The cricket model uses final match runs and wickets as input features. These are not available before a match starts, so the experiment should not be presented as a **pre-match** winner prediction system.
- The cricket dataset's origin and collection method are not documented in the provided files.
- The football labels are generated from a stamina threshold rather than actual injury data.
- In the football experiment, the same stamina feature used to define the label is also provided to the model.
- The football preprocessing fits the imputer before splitting the dataset, which can introduce data leakage.
- Both experiments use a single random train-test split; more careful validation would be needed before drawing general conclusions.

## Conclusion

This project helped me practice data preprocessing, classification models, performance comparison, and visualization using sports datasets. Among the three tested models, XGBoost performed best on the cricket dataset. The football experiment also showed how important it is to define the target properly and avoid using information that directly reveals the answer.

For future improvements, I would use only information available before a cricket match starts, validate on unseen matches, and use actual injury records for a meaningful football injury prediction task.
