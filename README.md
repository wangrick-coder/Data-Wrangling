---
title: Final Presentation - Student Mental Health and Depression
author: Will Angrick
format: html
---

As individuals, you will find data, generate questions, and give a presentation over your findings. Each person will be given 3 minutes to present.

Import data

```{python}
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
depression = pd.read_csv('~/Downloads/student_depression_dataset.csv')
```

Data Cleaning and Preparing: 

First, I checked and dropped any missing values. Then I used the extract function to pull out the first number found in the string in the "Sleep Duration" column and converted it from a string data type to float. Next, I converted the "Financial Stress" column from a string data type to float. Lastly, I create a numeric dataframe for analyses, only selecting numeric columns

```{python}
# Check for missing values
print('Missing values in each column:')
print(depression.isnull().sum())

# For simplicity, drop rows with missing values if there are any
depression2 = depression.dropna()

# With the format being 'Less than 5 hours', extract numeric value
depression2['Sleep Duration Numeric'] = depression2['Sleep Duration'].str.extract('(\d+)', expand=False).astype(float)

#Convert 'Financial Stress' data type from an object to a float
depression2["Financial Stress"] = pd.to_numeric(depression2["Financial Stress"], errors="coerce")

print('\nDataset Info after cleaning:')
print(depression2.info())

numeric_depression = depression2.select_dtypes(include=[np.number])
```

Visualization Data Analysis: 

How can I use visualizations to depict a well-rounded overview of the dataset and potentially reveal relationships between different variables and depression? 

I used a variety of visualizations, including histograms, bar plots, box plots and heatmaps to draw insights from the data and reveal interesting relationships that can easily be recognized. 

```{python}
plt.figure(figsize=(8, 6))
sns.histplot(depression2['Age'], kde=True, bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
```

Over 5,000 students are 20 years old or younger, while a majority of students fall between the ages of 20-30 years old.

```{python}
plt.figure(figsize=(6, 4))
sns.countplot(x='Depression', data=depression2, palette='pastel')
plt.title('Depression Count')
plt.xlabel('Depression Status')
plt.ylabel('Count')
plt.show()
```

More than 16,000 out of 27,901 students reported depression, representing about 57% of the dataset.


```{python}
plt.figure(figsize=(10, 8))
corr = numeric_depression.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.show()
```

This heatmap gives us a general understanding of the correlation between the numeric variables and depression rates as well as intervariable correlations. Academic pressure, financial stress, and work/study hours were most closely correlated with higher depression rates, while age and study satisfaction were most negatively correlated with depression. 

Question 1: How does sleep duration relate to depression rates?

```{python}
sleep_dep = depression2.groupby("Sleep Duration")["Depression"].mean().reset_index()
sleep_dep = sleep_dep.sort_values("Depression", ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(data=sleep_dep, x="Sleep Duration", y="Depression", palette='pastel')
plt.title("Depression Rate by Sleep Duration")
plt.xlabel("Sleep Duration")
plt.ylabel("Average Depression (0–1)")
plt.xticks(rotation=45)
plt.show()
sleep_dep
```

From this visualization, sleep duration generally has a negative downward relationship with depression rates. This means that as sleep duration increases, depression rates generally go down. Of the students that got less than 5 hours of sleep, 64.5% reported depression, while 51% of students that slept more than 8 hours reported depression representing over a 13% decrease. However, it is important to note that students who slept between 7-8 hours a night reported higher depression rates than those who slept for 5-6 hours.

Question 2: Which stressors are most closely related/predictive of depression?

```{python}
# Boxplot to visualize the academic pressure distribution for depressed and non-depressed students
plt.figure(figsize=(8,6))
sns.boxplot(x='Depression', y='Academic Pressure', data=depression2, palette='pastel')
plt.title('Academic Pressure by Depression Status')
plt.show()

# Boxplot to visualize the academic pressure distribution for depressed and non-depressed students
plt.figure(figsize=(8,6))
sns.boxplot(x='Depression', y='Financial Stress', data=depression2, palette='pastel')
plt.title('Financial Stress by Depression Status')
plt.show()
```

First, I examined the broader impacts of academic and financial stress by depression status. The median for students who did not report depression was two out of five for both academic and financial stress and four out of five for those who did report depression.


```{python}
stress_cols = ["Academic Pressure", "Work Pressure", "Financial Stress"]
# Average stress values for depressed vs non-depressed students
stress_dep = depression2.groupby("Depression")[stress_cols].mean().reset_index()
stress_long = stress_dep.melt(id_vars="Depression",
                              value_vars=stress_cols,
                              var_name="Stress Type",
                              value_name="Average Level")

plt.figure(figsize=(10,6))
sns.barplot(data=stress_long, x="Stress Type", y="Average Level", hue="Depression")
plt.title("Stress Levels by Depression Status")
plt.xlabel("Stress Factor")
plt.ylabel("Average Stress Level")
plt.legend(title="Depression", labels=["No", "Yes"])
plt.tight_layout()
plt.show()
```

Next, I used the melt function so I could turn the wide-format data from the 'stress_dep' subset into long format, which is the format required for grouped barplots.

```{python}
finstress_counts = depression2.groupby(["Financial Stress", "Depression"]).size().reset_index(name="Count")

plt.figure(figsize=(10,6))
sns.barplot(data=finstress_counts,
            x="Financial Stress",
            y="Count",
            hue="Depression",
            palette="pastel")

plt.title("Count of Students by Financial Stress Level and Depression Status")
plt.xlabel("Financial Stress Level")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()
```

```{python}
correlations = depression2[stress_cols + ["Depression"]].corr()
correlations
```

From these visualizations and correlation table, we can see that academic and financial stress are most closely related to depression. On the other hand, work pressure appears to have no correlation to depression as the survey population is students. Academic pressure is slightly more correlated to depression than financial stress with 0.48 and 0.36 correlations, respectively. As financial stress increases, we can see an increasing gap between students who reported depression and those who did not.


Question 3: Are students with suicidal thoughts more likely to report depression?

```{python}
suicide_dep = depression2.groupby("Have you ever had suicidal thoughts ?")["Depression"].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=suicide_dep,
            x="Have you ever had suicidal thoughts ?",
            y="Depression",
            palette="pastel")

plt.title("Depression Rate by Suicidal Thoughts")
plt.xlabel("Suicidal Thoughts")
plt.ylabel("Depression Rate")
plt.tight_layout()
plt.show()
```

```{python}
depression2["SuicideNumeric"] = depression2["Have you ever had suicidal thoughts ?"].map({
    "Yes": 1,
    "No": 0
})
depression2[["SuicideNumeric", "Depression"]].corr()

corr_subset = depression2[[
    "Sleep Duration Numeric",
    "Academic Pressure",
    "Financial Stress",
    "SuicideNumeric"
]].corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_subset, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: Sleep, Academic Pressure, Financial Stress, Suicidal Thoughts")
plt.tight_layout()
plt.show()
```

To no surprise, the depression rate for those with suicidal thoughts is roughly 80%, while it is below 25% for those without suicidal thoughts. However, I was curious to see how sleep duration, academic pressure, and financial stress correlated with suicidal thoughts. I was not surprised to see that academic pressure and financial stress were weakly correlated, while sleep duration had a weak negative correlation because there are many other factors such as job satisfaction, diet, and family history of mental illness that can lead to suicidal thoughts.


Simple Predictive Model:

I built a simple predictive model to determine whether a student is depressed based on the available features. I used a logistic regression classifier since the outcome 'Depression' is binary. In addition to generating the model, we compute the prediction accuracy on the test dataset. This approach provides useful insights into the relationship between the predictors and the outcome.

First, I selected the features I wanted to explore and created a model dataset using these predictors. Second, I needed to remove rows with NaN or missing/infinite values. Then set the X and y variables and added an intercept for my logistic regression model. 


```{python}
import statsmodels.api as sm

features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
            'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress',
            'Sleep Duration Numeric']

model_data = depression2[features + ['Depression']].copy()
model_data = model_data.replace([np.inf, -np.inf], np.nan).dropna()

X = model_data[features]
y = model_data['Depression']
X = sm.add_constant(X)

logit_model = sm.Logit(y, X).fit()
print(logit_model.summary())
```
