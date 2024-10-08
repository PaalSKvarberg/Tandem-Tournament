import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Function to infer round from variable name
def infer_round(variable_name):
    if 'pre_intervention' in variable_name and 'pre_teamdebate' in variable_name:
        return 'pre_predebate'
    elif 'pre_intervention' in variable_name and 'post_teamdebate' in variable_name:
        return 'pre_postdebate'
    elif 'post_intervention' in variable_name and 'pre_teamdebate' in variable_name:
        return 'post_predebate'
    elif 'post_intervention' in variable_name and 'post_teamdebate' in variable_name:
        return 'post_postdebate'
    return 'unknown'

# Directory containing different scenario files
data_directory = '/Users/Parnold/Dropbox/tetlock predictions/stan output'
# Flag to ensure data check is only reported once
data_check_reported = False

# Prepare a list to store results from all designs for scatter plot
all_designs_results = []
scenario_number = 1

# Loop through all CSV files in the directory
for file_name in os.listdir(data_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_directory, file_name)

        # Load data
        df = pd.read_csv(file_path)

        # Print intermediate checks for debugging
        print(f"Scenario {scenario_number}")
        print(f"Data shape: {df.shape}")

        # Add inferred round column
        df['round'] = df['variable'].apply(infer_round)

        # Validate counts with correct predictions per simulation
        actual_participants = df['participant'].nunique()
        actual_teams = df['team'].nunique()
        num_simulations = df['replication_index'].nunique()

        # Print the number of participants, teams, and simulations
        print(f"Number of simulations: {num_simulations}")
        print(f"Number of participants: {actual_participants}")
        print(f"Number of teams: {actual_teams}")

        # Collect results for all simulations
        all_simulation_results = []
        mean_scores = []
        mean_t_stat = []
        mean_p_val = []
        mean_cohens_d = []
        mean_power = []

        # Process each simulation separately
        for replication_index in df['replication_index'].unique():
            sim_df = df[df['replication_index'] == replication_index]

            # Process each of the 4 conditions separately
            conditions = ['pre_predebate', 'pre_postdebate', 'post_predebate', 'post_postdebate']
            condition_scores = {cond: sim_df[sim_df['round'] == cond]['brier'] for cond in conditions}
            mean_scores.append({cond: scores.mean() for cond, scores in condition_scores.items()})

            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*(condition_scores[cond] for cond in conditions))
            mean_t_stat.append(f_stat)
            mean_p_val.append(p_value)

            # Calculate the effects size (Cohen's f)
            pooled_var = sum(condition_scores[cond].var() * (len(condition_scores[cond])-1) for cond in conditions) / (sum(len(condition_scores[cond]) for cond in conditions) - len(conditions))
            mean_var = sum(condition_scores[cond].mean() for cond in conditions) / len(conditions)
            cohens_f = (sum((condition_scores[cond].mean() - mean_var)**2 / len(conditions) for cond in conditions) / pooled_var) ** 0.5
            mean_cohens_d.append(cohens_f)

            # Estimate the statistical power
            analysis = TTestIndPower()
            power = analysis.power(effect_size=cohens_f, nobs1=len(sim_df) // len(conditions), alpha=0.05, ratio=1.0)
            mean_power.append(power)

            # Store the results for this simulation
            simulation_result = {
                'replication_index': replication_index,
                'mean_brier_pre_predebate': mean_scores[-1]['pre_predebate'],
                'mean_brier_pre_postdebate': mean_scores[-1]['pre_postdebate'],
                'mean_brier_post_predebate': mean_scores[-1]['post_predebate'],
                'mean_brier_post_postdebate': mean_scores[-1]['post_postdebate'],
                'f_statistic': f_stat,
                'p_value': p_value,
                'cohens_f': cohens_f,
                'power': power,
                'design': file_name
            }
            all_simulation_results.append(simulation_result)

        # Append results from this design to the all designs list
        all_designs_results.extend(all_simulation_results)

        # Print aggregated results for this scenario
        print("\n  Mean Brier Scores by Condition:", mean_scores[-1])
        print("  Mean F-statistic:", sum(mean_t_stat) / len(mean_t_stat))
        print("  Mean P-value:", sum(mean_p_val) / len(mean_p_val))
        print("  Mean Cohen's f:", sum(mean_cohens_d) / len(mean_cohens_d))
        print("  Mean Power:", sum(mean_power) / len(mean_power), "\n")

        scenario_number += 1

# Create a DataFrame from the combined results of all designs
all_designs_df = pd.DataFrame(all_designs_results)

# Visualize results from all designs in a scatter plot with regression lines
plt.figure(figsize=(14, 10))
for condition in ['pre_predebate', 'pre_postdebate', 'post_predebate', 'post_postdebate']:
    sns.scatterplot(data=all_designs_df, x='replication_index', y=f'mean_brier_{condition}', label=condition, alpha=0.5)
    sns.regplot(data=all_designs_df, x='replication_index', y=f'mean_brier_{condition}', scatter=False, label=f'{condition} Regression Line')

plt.legend(loc='upper right')
plt.title('Scatter Plot of Mean Brier Scores for Different Conditions with Regression Lines')
plt.xlabel('Replication Index')
plt.ylabel('Mean Brier Score')
plt.show()

# Calculate the number of participants needed for sufficient power
desired_power = 0.8
alpha = 0.05
mean_effect_size = all_designs_df['cohens_f'].mean()
analysis = TTestIndPower()
required_n = analysis.solve_power(effect_size=mean_effect_size, power=desired_power, alpha=alpha, ratio=1.0)

print(f"\nThe number of participants required to achieve a power of {desired_power} with an alpha of {alpha} and an average effect size {mean_effect_size:.2f} is approximately {required_n:.0f}.")
