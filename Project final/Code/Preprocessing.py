("***PREPROCESSING***")
#Authors: Ruqia Ali Hassan, Joelle Schiffmann, Lara Jenni

#importing libraries
from imblearn.over_sampling import RandomOverSampler, SMOTE
import math
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import scipy.stats as sts
from scipy.stats import shapiro, chi2_contingency
import seaborn as sns
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mutual_info_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
"""
#showing all data in tabel
pd.set_option('display.max_columns', None)
"""

("***DATA PREPROCESSING AND VISUALIZATION***")
#loading data
data = pd.read_csv("../data/dataset.csv")

#shape
print("shape ", data.shape)

#columns
print(data.columns)

#correcting feature names
data.rename({'baseline value':'baseline_value',
                    'prolongued_decelerations':'prolonged_decelerations',
                    'abnormal_short_term_variability':'pc_short_term_ab_variability',
                    'mean_value_of_short_term_variability':'mean_short_term_variability',
                    'percentage_of_time_with_abnormal_long_term_variability':'pc_long_term_ab_variability',
                    'mean_value_of_long_term_variability':'mean_long_term_variability',
                    'histogram_number_of_peaks':'histogram_peak_count',
                    'histogram_number_of_zeroes':'histogram_zero_count'
                    }, axis = 1, inplace = True)

#unique values
for feature in data.columns:
    print("Feature:", feature)
    unique_values = data[feature].unique()
    print("unique values:", unique_values)
    print("Sum of unique values:", unique_values.shape[0])
    print("")

#data types
print(data.dtypes)

#chaning data types
data["baseline_value"] = data["baseline_value"].astype("int32")
data["histogram_width"] = data["histogram_width"].astype("int32")
data["histogram_min"] = data["histogram_min"].astype("int32")
data["histogram_max"] = data["histogram_max"].astype("int32")
data["histogram_peak_count"] = data["histogram_peak_count"].astype("int32")
data["histogram_zero_count"] = data["histogram_zero_count"].astype("int32")
data["histogram_mode"] = data["histogram_mode"].astype("int32")
data["histogram_tendency"] = pd.Categorical(data["histogram_tendency"])
data["fetal_health"] = pd.Categorical(data["fetal_health"])

#lists for categorical and numerical features
num_cols = []
cat_cols = []
for col in data.columns:
    if data[col].dtype in ["float", "int"]:
        num_cols.append(col)
    else:
        cat_cols.append(col)
print("numerical features: ", num_cols)
print("categorical features: ", cat_cols)

#head
print(data.head(5))

#missing data
print("missing values ", data.isna().sum(axis=1).sum())

#duplicated data
data.duplicated().sum()
print("duplicated rows ", data.duplicated().sum())
data=data.drop_duplicates(keep="first")
print("shape cleaned data ", data.shape)

#data summary statistics
print(data.describe(include = "all"))

#color blind palette
sns.set_palette("colorblind")

#units
units = [
    "beats per minute (bpm)",  # Baseline Value
    "accelerations per second",  # Accelerations
    "movements per second",  # Fetal Movement
    "contractions per second",  # Uterine Contractions
    "decelerations per second",  # Light Decelerations
    "decelerations per second",  # Severe Decelerations
    "decelerations per second",  # Prolonged Decelerations
    "percent (%)",  # Percentage of Time with Abnormal Short Term Variability
    "milliseconds (ms)",  # Mean Value of Short Term Variability
    "percent (%)",  # Percentage of Time with Abnormal Long Term Variability
    "milliseconds (ms)",  # Mean Value of Long Term Variability
    "milliseconds (ms)",  # Histogram Width
    "milliseconds (ms)",  # Histogram Min
    "milliseconds (ms)",  # Histogram Max
    "number of peaks",  # Histogram Peak Count
    "number of zeros",  # Histogram Zero Count
    "milliseconds (ms)",  # Histogram Mode
    "milliseconds (ms)",  # Histogram Mean
    "milliseconds (ms)",  # Histogram Median
    "square milliseconds (ms^2)",  # Histogram Variance
    "categorical",  # Histogram Tendency
    "categorical"  # Fetal Health
]

#histograms for numerical features
num_features = len(num_cols)
num_rows = math.ceil(num_features / 4)
num_cols_subplot = min(num_features, 4)

plt.figure(figsize=(20, num_rows * 5))
for i, label in enumerate(num_cols):
    plt.subplot(num_rows, num_cols_subplot, i+1)
    for fetal_health, health_label in zip([1, 2, 3], ["normal", "suspect", "pathological"]):
        plt.hist(data[data["fetal_health"] == fetal_health][label], label=health_label, alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Density")
    plt.xlabel(units[i])
    plt.legend()
plt.tight_layout()
plt.savefig("../output/histograms_numerical.png")
plt.show()

#countplots for categorical features
fig, axes = plt.subplots(1, len(cat_cols), figsize=(5*len(cat_cols), 5))
for i, col in enumerate(cat_cols):
    sns.countplot(x=col, data=data, ax=axes[i])
    axes[i].set_title(f'Countplot for {col}')
    for p in axes[i].patches:
        axes[i].annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.tight_layout()
plt.savefig("../output/countplots_categorical.png")
plt.show()
#target fetal health is imbalanced

#boxplots for outliers
num_features = len(num_cols)
num_rows = math.ceil(num_features / 2)

fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5*num_rows))

for i, label in enumerate(num_cols):
    row = i // 2
    col = i % 2
    sns.boxplot(x=data["fetal_health"], y=data[label], hue=data["fetal_health"].map({1: "normal", 2: "suspect", 3: "pathological"}), ax=axes[row, col])
    axes[row, col].set_title(f'Boxplots for {label}')
    axes[row, col].set_xlabel('Fetal Health')
    axes[row, col].set_ylabel(units[i])
    axes[row, col].legend(title="Fetal Health")
    axes[row, col].xaxis.set_ticklabels([])
plt.tight_layout()
plt.savefig("../output/boxplots_outliers.png")
plt.show()
#outliers all seemed medically possible

("***STATISTICAL TESTING***")
#normality with Shapiro-Wilk Test (only numerical features)
for col in num_cols:
    stat, p = shapiro(data[col])
    print(f"Feature: {col}")
    print(f"Shapiro-Wilk Test Statistic: {stat}, p-value: {p}")
    if p < 0.05:
        print("The feature is not normally distributed.")
    else:
        print("The feature is normally distributed.")
    print("")
#all numerical features are not normally distributed

#normality with QQplots (only numerical features)
plt.figure(figsize=(15, 10))
for i, feature in enumerate(num_cols, 1):
    plt.subplot(5, 4, i)
    sts.probplot(data[feature], dist="norm", plot=plt)
    plt.title(f'QQ-Plot for {feature}')
plt.tight_layout()
plt.savefig("../output/continuous_features_qqplots.png")
plt.show()
#baseline_value, histogram_median, histogram_max, histogram_mean look normally distributed
num_cols_norm = ['baseline_value', 'histogram_median', 'histogram_max', 'histogram_mean']
num_cols_not_norm = [col for col in num_cols if col not in num_cols_norm]

#Kruskal-Wallis H-Test (not normally distributed)
alpha = 0.05 / (len(data.columns)-1)  # bonferroni correction
print("alpha = ", alpha)
for var in num_cols_not_norm:
    sample = [data[var][data["fetal_health"] == cat] for cat in data["fetal_health"].unique()]
    kruskal_stat, kruskal_p = sts.kruskal(*sample)
    print(f"Kruskal-Wallis H test for {var}:")
    print(f"Kruskal-Wallis: {kruskal_stat}")
    print(f"P-value: {kruskal_p}")

    if kruskal_p < alpha:
        print(f"There is a significant difference between {var} and fetal_health")
    else:
        print(f"No significant difference between {var} and fetal_health")
    print("\n")
#No significant diffrence: histogram_zero_count

#ANOVA (normally distributed)
alpha = 0.05 / (len(data.columns)-1)  # Bonferroni correction
print("alpha = ", alpha)
for var in num_cols_norm:
    sample = [data[var][data["fetal_health"] == cat] for cat in data["fetal_health"].unique()]
    anova_stat, anova_p = sts.f_oneway(*sample)
    print(f"ANOVA for {var}:")
    print(f"ANOVA F-statistic: {anova_stat}")
    print(f"P-value: {anova_p}")

    if anova_p < alpha:
        print(f"There is a significant difference between {var} and fetal_health")
    else:
        print(f"No significant difference between {var} and fetal_health")
    print("\n")
#No significant diffrence: histogram_max

#Chi sqaure (categorical features)
target = ['fetal_health']
cat_cols_without_target = [col for col in cat_cols if col not in target]
alpha = 0.05 / (len(data.columns)-1)  # bonferroni correction
print("alpha = ", alpha)
for col in cat_cols_without_target:
    contingency_table = pd.crosstab(data[col], data['fetal_health'])
    chi2_stat, chi2_p, dof, ex = chi2_contingency(contingency_table)
    print(f"Chi-square Test for {col}:")
    print(f"Chi-square Statistic: {chi2_stat:.10f}, p = {chi2_p:.10f}")
    if chi2_p < alpha:
        print(f"There is a significant association between {col} and fetal_health.\n")
    else:
        print(f"No significant association between {col} and fetal_health.\n")

#feature correlation (categorical-categorical: Cramér's, numerical-numerical: Spearman's, categorical-numerical: Kendall's)
# empty matrix with len(data.columns)
num_features = len(data.columns)
correlation_matrix = np.zeros((num_features, num_features))
#cramers function
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2_stat, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2_stat / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
#going through all features
for i, feature1 in enumerate(data.columns):
    for j, feature2 in enumerate(data.columns):
        if i == j:
            correlation_matrix[i, j] = 1
        else:
            #categorical-categorical
            if data[feature1].dtype == 'category' and data[feature2].dtype == 'category':
                #Cramér's V-Test
                cramers_v_stat = cramers_v(data[feature1], data[feature2])
                correlation_matrix[i, j] = cramers_v_stat
            #numerical - numerical
            if data[feature1].dtype in ['int', 'float'] and data[feature2].dtype in ['int', 'float']:
                #Spearman's Rank correlation test
                spearman_corr, _ = sts.spearmanr(data[feature1], data[feature2])
                correlation_matrix[i, j] = spearman_corr
            #categorical-numerical
            if (data[feature1].dtype == 'category' and data[feature2].dtype in ['int', 'float']) or \
                    (data[feature2].dtype == 'category' and data[feature1].dtype in ['int', 'float']):
                #Kendall's Tau correlation test
                kendall_corr, _ = sts.kendalltau(data[feature1], data[feature2])
                correlation_matrix[i, j] = kendall_corr
columns = data.columns
df_matrix = pd.DataFrame(correlation_matrix, columns=columns, index=columns)
print(df_matrix)
#heat map
plt.figure(figsize=(20, 20))
sns.heatmap(df_matrix, annot=True, cmap="coolwarm", linewidths=.5, cbar=True, center=0)
plt.title("Feature Correlation Heatmap (Cramér's, Kendall's, Spearman's)")
plt.tight_layout()
plt.savefig("../output/heatmap_correlation.png")
plt.show()
#<|0.1|: 'light_decelerations,' 'histogram_max,' 'histogram_peak_count,' 'histogram_zero_count,' 'histogram_mode,' 'histogram_mean,' 'histogram_median'
#>|0.3|: 'accelerations,' 'prolonged_decelerations,' 'pc_short_term_ab_variability,' 'pc_long_term_ab_variability'

("***DATA SPLITTING & SCALING***")
#label encoding (target)
y = data['fetal_health']
y_encoded = LabelEncoder().fit_transform(y)
print(y_encoded)

#one hot encoding (features)
X = data[data.columns.drop(['fetal_health'])]
X_encoded = pd.get_dummies(data, columns=['histogram_tendency'], prefix='histogram_tendency', drop_first=False, dtype = int)
print(X_encoded)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1111, random_state=42)
print('Training set size: {}, validation set size: {}, test set size: {}'.format(len(X_train), len(X_valid), len(X_test)))

#scaling
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
X_valid[num_cols] = scaler.transform(X_valid[num_cols])

#balancing training data with SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

#missing values after SMOTE
print("Checking for NaNs in the dataset...")
print(X_train_balanced.isna().sum().sum())
def find_columns_with_nans(data):
    columns_with_nans = data.columns[data.isna().any()].tolist()
    return columns_with_nans
columns_with_nans = find_columns_with_nans(X_train_balanced)
print("Columns with NaN values:", columns_with_nans)
#missing values in histogram_tendency

#replacing missing values in histogram_tendency
imputer = SimpleImputer(strategy='most_frequent')
feature_with_missing_values = ['histogram_tendency']
X_train_balanced[feature_with_missing_values] = imputer.fit_transform(X_train_balanced[feature_with_missing_values])

#rechecking for missing values after replacing missing values in histogram_tendency
print("Checking for NaNs in the dataset...")
print(X_train_balanced.isna().sum().sum())
def find_columns_with_nans(data):
    columns_with_nans = data.columns[data.isna().any()].tolist()
    return columns_with_nans
columns_with_nans = find_columns_with_nans(X_train_balanced)
print("Columns with NaN values:", columns_with_nans)
#no missing values


# Create a count plot for the balanced training data
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=y_train_balanced)

# Add annotations to the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 10), textcoords='offset points')


plt.title('Distribution of Fetal Health Categories After SMOTE-training data')
plt.xlabel('Fetal Health Category')
plt.ylabel('Count')
plt.show()

#all categories have same amount 1309
