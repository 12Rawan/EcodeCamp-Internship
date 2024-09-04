import matplotlib.pyplot as plt
import seaborn as sns


# Histogram for Age
def plot_age_distribution(df_train, save_as='age_distribution.png'):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_train['Age'], kde=True, bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig(save_as)
    plt.close()


# Count pot for genders who survived
def plot_survival_by_sex(df_train, save_as='survival_by_sex.png'):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Survived', hue='Sex', data=df_train)
    plt.title('Survival by Sex')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.savefig(save_as)
    plt.close()


# Correlation heatmap
# to see how different the features correlate to each others and with the target variable
def plot_correlation_heatmap(df_train, save_as='correlation_heatmap.png'):
    plt.figure(figsize=(12, 8))
    corr = df_train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
                     'Survived']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.savefig(save_as)
    plt.close()


# Count the number of survived people and non-survived people
# Create a bar plot
def plot_survival_counts(df_train, save_as='survival_counts.png'):
    survival_counts = df_train['Survived'].value_counts()
    plt.figure(figsize=(8, 6))
    sns.barplot(x=survival_counts.index,
                y=survival_counts.values,
                palette='viridis')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.title('Count of Survived vs Not Survived')
    plt.xticks(ticks=[0, 1], labels=['Not Survived', 'Survived'])
    plt.savefig(save_as)
    plt.close()
