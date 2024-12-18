import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.stats import gaussian_kde
import matplotlib.patheffects as path_effects




# ---
# ---
# 
# # CLEANING


# ---
# 
# ### DESCRIBE


df = pd.read_csv('data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')


df.isna().sum()


# ---
# 
# ### CLEAN


df['TotalCharges'] = df['TotalCharges'].replace({' ':'0'})
df['TotalCharges'] = df['TotalCharges'].astype(float)/df['tenure']

df = df.fillna(0.)


# ---
# 
# ### DUMMIFY


to_replace = {
    'No' : 0,
    'Yes': 1
}

for c in ['Partner','Dependents','PhoneService', 'PaperlessBilling', 'Churn']:
    df[c] = df[c].map(to_replace)


to_replace = {
    'Female' : 0,
    'Male' : 1
}

df['gender'] = df['gender'].map(to_replace)


to_replace = {
    'No phone service': 0,
    'No': 1,
    'Yes': 2
}

df['MultipleLines'] = df['MultipleLines'].map(to_replace)


to_replace = {
    'DSL' : 1,
    'Fiber optic': 2,
    'No' : 0
}

df['InternetService'] = df['InternetService'].map(to_replace)


to_replace = {
    'No' : 0,
    'Yes': 1,
    'No internet service' : 0
}

for c in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[c] = df[c].map(to_replace)


to_replace = {
    'Month-to-month': 0,
    'One year': 1,
    'Two year': 2
}

df['Contract'] = df['Contract'].map(to_replace)


df = pd.get_dummies(df, columns=['PaymentMethod'], prefix='PaymentMethod', drop_first=True, dtype=int)


# ---
# ---
# 
# # COHORT ANALYSIS


df['tenure_cohort'] = pd.cut(list(df['tenure']),[0, 12, 24, float('inf')], labels=['0 - 12', '12 - 24', '25+'])


cohort_data = df.groupby('tenure_cohort')['Churn'].mean()
cohort_data


# ---
# ---
# 
# # LTV MODEL 


# ---
# 
# ### LIFETIME PREDICTION WITH REGRESSION
# 
# Lifetimes are known for churned clients: by training a regressor model on this cohort, we can then predict the total tenure of other clients, hence their LTV


def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)

    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


df_churned = df.loc[df['Churn'] == 1]


X = df_churned.drop(['Churn','tenure'],axis=1).select_dtypes(['number'])
y = df_churned['tenure']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)


model = RandomForestRegressor()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)



regression_results(y_test,y_pred)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [3,5,7,8,10],
    'min_samples_leaf': [ 4, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=3, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)



best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

regression_results(y_test,y_pred)


# ---
# 
# ### SURVIVAL ANALYSIS
# 
# Survival analysis does not require to remove customers that didn't churn, so it's more robust


kmf = KaplanMeierFitter()

# Fit the model
kmf.fit(df['tenure'], event_observed=df['Churn'])

# Plot survival function
kmf.plot_survival_function()
plt.title('Survival Function')
plt.show()

predicted_survival = kmf.survival_function_at_times([12, 24, 36, 48, 60, 72])
predicted_survival


cph = CoxPHFitter()

# Select relevant features and fit the model
cph.fit(df.select_dtypes(['number']), 'tenure', event_col='Churn')

# View results
cph.print_summary()


# ---
# 
# ### LTV COMPUTATION


active_customers = df[df['Churn'] == 0].copy()

active_customers['expected_lifetime_cph'] = cph.predict_expectation(active_customers)
active_customers['expected_lifetime_rf'] = model.predict(active_customers.select_dtypes(['number']).drop(['Churn','expected_lifetime_cph','tenure'],axis=1))


active_customers['LTV_cph'] = active_customers['expected_lifetime_cph']*active_customers['TotalCharges']
active_customers['LTV_rf'] = active_customers['expected_lifetime_rf']*active_customers['TotalCharges']



# Extract LTV values for the two models
ltv_cph = active_customers['LTV_cph']
ltv_rf = active_customers['LTV_rf']
ltv_churned = df_churned['TotalCharges']*df_churned['tenure']

bwm = 0.25

# Define KDE for both distributions
kde_cph = gaussian_kde(ltv_cph, bw_method=bwm)  # Adjust `bw_method` for smoothing level
kde_rf = gaussian_kde(ltv_rf, bw_method=bwm)
kde_churned = gaussian_kde(ltv_churned, bw_method=bwm)

# Create a range of x values for smooth plotting
x_range = np.linspace(min(ltv_cph.min(), ltv_rf.min()), max(ltv_cph.max(), ltv_rf.max()), 500)

# Evaluate KDE for both models
kde_cph_values = kde_cph(x_range)
kde_rf_values = kde_rf(x_range)
kde_churned_values = kde_churned(x_range)

# Plot the KDE distributions
plt.figure(figsize=(10, 6))

plt.plot(x_range, kde_cph_values, label='Cox Proportional Hazards Model', color='C0', linewidth=2, alpha=0.8)
plt.plot(x_range, kde_rf_values, label='Random Forest', color='C1', linewidth=2, alpha=0.8)
plt.plot(x_range, kde_churned_values, label='Churned Actual Value', color='C3', linewidth=2, alpha=0.8)

# Add labels, legend, and formatting
plt.fill_between(x_range, kde_cph_values, alpha=0.7, color='C0')  # Add shaded area under the curve
plt.fill_between(x_range, kde_rf_values, alpha=0.7, color='C1')
plt.fill_between(x_range, kde_churned_values, alpha=0.3, color='C3')

plt.title('Smoothed LTV Distributions (Gaussian KDE)', fontsize=14)
plt.xlabel('LTV', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# Remove the top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add gridlines only on the y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add white outlines to all text elements (axes labels, ticks, and titles)
ax = plt.gca()
texts = [ax.title, ax.xaxis.label, ax.yaxis.label]  # Title, xlabel, ylabel
texts += ax.get_xticklabels() + ax.get_yticklabels()  # Tick labels

# Apply white outlines to each text object
for text in texts:
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
        path_effects.Normal()  # Normal text rendering
    ])

# Tight layout and save as a transparent image
plt.tight_layout()
plt.savefig('artifacts/imgs/predicted_LTV.png', transparent=True)

# Display the plot
plt.show()


# ---
# ---
# 
# # LTV ANALYSIS


# ---
# 
# ### LTV SEGMENTS


active_customers['LTV_segments'] = pd.cut(active_customers['LTV_cph'],bins=[0,3000,6000,100000],labels=['Bronze','Silver','Gold'])


ltv_segments_recap = active_customers.groupby('LTV_segments').size().reset_index(name='count')
ltv_segments_recap['percentage'] = (ltv_segments_recap['count'] / ltv_segments_recap['count'].sum()) * 100
ltv_segments_recap


# Colors for the segments
colors = {'Bronze': '#AD4A13', 'Silver': '#E4E4E4', 'Gold': '#FFCC00'}

# Plot the data
plt.figure(figsize=(8, 6))

# Create the bar plot with a modern style
bars = plt.bar(
    ltv_segments_recap['LTV_segments'], 
    ltv_segments_recap['percentage'], 
    color=[colors[segment] for segment in ltv_segments_recap['LTV_segments']],
    edgecolor='black',  # Add a black border for better contrast
    linewidth=1,
    zorder=10
)

# Add percentage labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Center horizontally
        height + 0.5,  # Slightly above the bar
        f'{height:.1f}%', 
        ha='center', 
        va='bottom', 
        fontsize=10, 
        color='black', 
        fontweight='bold',
        zorder=15,
        path_effects=[
            path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
            path_effects.Normal()  # Normal text rendering
        ]
    )

# Add titles and labels
plt.title('LTV Segments Recap', fontsize=18, pad=15, fontweight='bold')
plt.xlabel('LTV Segments', fontsize=14, labelpad=10)
plt.ylabel('Percentage (%)', fontsize=14, labelpad=10)

# Remove the top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add gridlines only on the y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add white outlines to all text elements (axes labels, ticks, and titles)
ax = plt.gca()
texts = [ax.title, ax.xaxis.label, ax.yaxis.label]  # Title, xlabel, ylabel
texts += ax.get_xticklabels() + ax.get_yticklabels()  # Tick labels

# Apply white outlines to each text object
for text in texts:
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
        path_effects.Normal()  # Normal text rendering
    ])

# Tight layout and save as a transparent image
plt.tight_layout()
plt.savefig('artifacts/imgs/LTV_segments_recap.png', transparent=True, dpi=300)

# Display the plot
plt.show()


df_cluster = pd.read_csv('data/raw/telco_clusters.csv')

active_customers = pd.merge(active_customers,df_cluster[['cluster','cluster_name']],left_index=True, right_index=True, how='left')


ltv_cluster_recap = active_customers.groupby(['cluster_name','LTV_segments']).size().reset_index(name='count')

ltv_cluster_recap['cluster_total'] = ltv_cluster_recap.groupby('cluster_name')['count'].transform('sum')

# Calculate the percentage for each segment within the cluster
ltv_cluster_recap['percentage'] = (ltv_cluster_recap['count'] / ltv_cluster_recap['cluster_total']) * 100

# Drop the cluster_total column if not needed
ltv_cluster_recap.drop(columns=['cluster_total'], inplace=True)

# View the result
ltv_cluster_recap


segment_colors = {'Bronze': '#AD4A13', 'Silver': '#E4E4E4', 'Gold': '#FFCC00'}

# Pivot the DataFrame to make it suitable for stacked bar plotting
ltv_cluster_pivot = ltv_cluster_recap.pivot(index='cluster_name', columns='LTV_segments', values='count').fillna(0)
ltv_percentage_pivot = ltv_cluster_recap.pivot(index='cluster_name', columns='LTV_segments', values='percentage').fillna(0)

# Plot each LTV segment as a stacked layer
fig, ax = plt.subplots(figsize=(9, 7))
bottom = None

# Loop through each segment and create stacked bars
for segment in ltv_cluster_pivot.columns:
    percentage_in_cluster = list(ltv_percentage_pivot[segment])
    bars = ax.bar(
        ltv_cluster_pivot.index, 
        ltv_cluster_pivot[segment], 
        label=segment, 
        color=segment_colors[segment], 
        bottom=bottom,
        edgecolor='black',  # Add a black border for better contrast
        linewidth=1,
        zorder=10
    )

    bottom = ltv_cluster_pivot[segment] if bottom is None else bottom + ltv_cluster_pivot[segment]
    
    # Add annotations (count and percentage) on each bar segment
    for i,bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:  # Avoid annotating zero values
            # Calculate the percentage for this segment
            cluster_name = bar.get_x() + bar.get_width() / 2  # Get x position of the bar
            percentage = percentage_in_cluster[i]

            # Annotate the count and percentage (1 decimal place)
            # ax.text(
            #     bar.get_x() + bar.get_width() / 2, 
            #     bar.get_y() + height / 2, 
            #     f'{percentage:.1f}%', 
            #     ha='center', 
            #     va='center', 
            #     fontsize=10, 
            #     color='black',
            #     zorder=15
            # )

            # Add annotations with a white outline around the text
            ax.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_y() + height / 2, 
                f'{percentage:.1f}%', 
                ha='center', 
                va='center', 
                fontsize=10, 
                color='black',  # Text color
                zorder=15,
                path_effects=[
                    path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
                    path_effects.Normal()  # Normal text rendering
                ]
            )

# Add titles and labels
ax.set_title('LTV Segments by Cluster', fontsize=18, pad=20)
ax.set_xlabel('Cluster Name', fontsize=14, labelpad=10)
ax.set_ylabel('Count', fontsize=14, labelpad=10)

# Customize ticks
ax.set_xticks(range(len(ltv_cluster_pivot.index)))
ax.set_xticklabels(ltv_cluster_pivot.index, rotation=45, ha='right', fontsize=12)

# Remove the top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add gridlines only on the y-axis for readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add white outlines to all text elements (axes labels, ticks, and titles)
ax = plt.gca()
texts = [ax.title, ax.xaxis.label, ax.yaxis.label]  # Title, xlabel, ylabel
texts += ax.get_xticklabels() + ax.get_yticklabels()  # Tick labels

# Apply white outlines to each text object
for text in texts:
    text.set_path_effects([
        path_effects.Stroke(linewidth=2, foreground='white'),  # White outline
        path_effects.Normal()  # Normal text rendering
    ])

# Add a legend
ax.legend(title='LTV Segments', title_fontsize=12, fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))

# Tight layout and save as a transparent image
plt.tight_layout()
plt.savefig('artifacts/imgs/LTV_clusters_recap.png', transparent=True)

# Display the plot
plt.show()



