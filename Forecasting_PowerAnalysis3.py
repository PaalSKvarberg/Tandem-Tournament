import pandas as pd
import numpy as np
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

# Pre-defined scenario information
scenario_data = {
    1: {"pre_intervention_question_count": 30, "post_intervention_question_count": 30, "antall_team": 60, "antall_medlemmer_per_team": 10},
    2: {"pre_intervention_question_count": 20, "post_intervention_question_count": 20, "antall_team": 60, "antall_medlemmer_per_team": 10},
    3: {"pre_intervention_question_count": 20, "post_intervention_question_count": 20, "antall_team": 25, "antall_medlemmer_per_team": 10},
    4: {"pre_intervention_question_count": 90, "post_intervention_question_count": 90, "antall_team": 60, "antall_medlemmer_per_team": 5},
    5: {"pre_intervention_question_count": 30, "post_intervention_question_count": 30, "antall_team": 180, "antall_medlemmer_per_team": 5}
}

# Directory containing different scenario files
data_directory = '/Users/Parnold/Dropbox/tetlock predictions/stan output'

# Prepare a list to store results from all designs for scatter plot
all_designs_results = []
scenario_number = 1

# Loop through all CSV files in the directory
for file_name in os.listdir(data_directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_directory, file_name)

        # Load data
        df = pd.read_csv(file_path)

        # Add inferred round column
        df['round'] = df['variable'].apply(infer_round)

        # Validate counts with correct predictions per simulation
        actual_participants = df['participant'].nunique()
        actual_teams = df['team'].nunique()
        num_simulations = df['replication_index'].nunique()

        # Scenario specific data
        scenario_info = scenario_data.get(scenario_number, {})
        num_teams = scenario_info.get("antall_team", "Unknown")
        members_per_team = scenario_info.get("antall_medlemmer_per_team", "Unknown")
        pre_intervention_questions = scenario_info.get("pre_intervention_question_count", "Unknown")
        post_intervention_questions = scenario_info.get("post_intervention_question_count", "Unknown")

        # Print the number of participants, teams, and simulations
        print(f"Scenario {scenario_number}")
        print(f"Data shape: {df.shape}")
        print(f"Number of simulations: {num_simulations}")
        print(f"Number of participants: {actual_participants}")
        print(f"Number of teams: {num_teams}")
        print(f"Number of participants on each team: {members_per_team}")
        print(f"Number of questions pre intervention: {pre_intervention_questions}")
        print(f"Number of questions post intervention: {post_intervention_questions}\n")

        # Collect results for all simulations
        all_simulation_results = []
        mean_scores = []
        mean_t_stat = []
        mean_p_val = []
        mean_cohens_f = []
        mean_power = []
        mean_n_required_power_0_8_f = []

        # Additional lists for Cohen's d power analysis
        mean_cohens_d_pre_post = []
        mean_power_cohens_d_pre_post = []
        mean_t_stat_pre_post = []
        mean_p_val_pre_post = []
        mean_power_pre_post = []
        mean_n_required_power_0_8_pre_post = []

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
            cohens_f = (sum((condition_scores[cond].mean() - mean_var)**2 for cond in conditions) / pooled_var) ** 0.5
            mean_cohens_f.append(cohens_f)

            # Estimate the statistical power
            analysis = TTestIndPower()
            power = analysis.power(effect_size=cohens_f, nobs1=len(condition_scores[conditions[0]]), alpha=0.05, ratio=1.0)
            mean_power.append(power)

            # Calculate required sample size to achieve a power of 0.8 for Cohen's f
            n_required_power_0_8_f = analysis.solve_power(effect_size=cohens_f, power=0.8, alpha=0.05)
            mean_n_required_power_0_8_f.append(n_required_power_0_8_f)

            # Calculate Cohen's d between pre_predebate and post_predebate
            mean_diff_pre_post = condition_scores['pre_predebate'].mean() - condition_scores['post_predebate'].mean()
            pooled_sd_pre_post = ((condition_scores['pre_predebate'].std()**2 + condition_scores['post_predebate'].std()**2) / 2) ** 0.5
            cohens_d_pre_post = mean_diff_pre_post / pooled_sd_pre_post
            mean_cohens_d_pre_post.append(cohens_d_pre_post)

            # Perform power analysis for Cohen's d
            power_cohens_d_pre_post = analysis.power(effect_size=cohens_d_pre_post, nobs1=len(condition_scores['pre_predebate']), alpha=0.05, ratio=1.0)
            mean_power_cohens_d_pre_post.append(power_cohens_d_pre_post)

            # Perform T-test for Pre_predebate vs Post_predebate
            t_stat_pre_post, p_val_pre_post = stats.ttest_ind(condition_scores['pre_predebate'], condition_scores['post_predebate'])
            mean_t_stat_pre_post.append(t_stat_pre_post)
            mean_p_val_pre_post.append(p_val_pre_post)

            # Power analysis for the T-test (also same as Cohen's d)
            power_pre_post = analysis.power(effect_size=cohens_d_pre_post, nobs1=len(condition_scores['pre_predebate']), alpha=0.05)
            mean_power_pre_post.append(power_pre_post)

            # Calculate sample size needed to achieve a power of 0.8 for Cohen's d
            sample_size_needed_pre_post = analysis.solve_power(effect_size=cohens_d_pre_post, power=0.8, alpha=0.05, ratio=1.0)
            mean_n_required_power_0_8_pre_post.append(sample_size_needed_pre_post)

            # Store the results for this simulation sequence
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
                'cohens_d_pre_post': cohens_d_pre_post,
                'power_cohens_d_pre_post': power_cohens_d_pre_post,
                't_statistic_pre_post': t_stat_pre_post,
                'p_value_pre_post': p_val_pre_post,
                'power_pre_post': power_pre_post,
                'n_required_power_0_8_pre_post': sample_size_needed_pre_post,
                'n_required_power_0_8_f': n_required_power_0_8_f,
                'design': file_name
            }
            all_simulation_results.append(simulation_result)

        # Append results from this design to the all designs list
        all_designs_results.extend(all_simulation_results)

        # Print aggregated results for this scenario
        aggregated_mean_scores = {cond: sum(score[cond] for score in mean_scores) / len(mean_scores) for cond in conditions}
        aggregated_mean_t_stat = sum(mean_t_stat) / len(mean_t_stat)
        aggregated_mean_p_val = sum(mean_p_val) / len(mean_p_val)
        aggregated_mean_cohens_f = sum(mean_cohens_f) / len(mean_cohens_f)
        aggregated_mean_power = sum(mean_power) / len(mean_power)
        aggregated_mean_n_required_power_0_8_f = sum(mean_n_required_power_0_8_f) / len(mean_n_required_power_0_8_f)
        aggregated_mean_cohens_d_pre_post = sum(mean_cohens_d_pre_post) / len(mean_cohens_d_pre_post)
        aggregated_mean_power_cohens_d_pre_post = sum(mean_power_cohens_d_pre_post) / len(mean_power_cohens_d_pre_post)
        aggregated_mean_t_stat_pre_post = sum(mean_t_stat_pre_post) / len(mean_t_stat_pre_post)
        aggregated_mean_p_val_pre_post = sum(mean_p_val_pre_post) / len(mean_p_val_pre_post)
        aggregated_mean_power_pre_post = sum(mean_power_pre_post) / len(mean_power_pre_post)
        aggregated_mean_n_required_power_0_8_pre_post = sum(mean_n_required_power_0_8_pre_post) / len(mean_n_required_power_0_8_pre_post)

        print("\nScenario Aggregated Results:")
        print("Mean Brier Scores by Condition Across all Simulations:")
        print(f'Pre_predebate = "Pre Intervention, before group discussion": {aggregated_mean_scores["pre_predebate"]}')
        print(f'Pre_postdebate = "Pre Intervention, after group discussion": {aggregated_mean_scores["pre_postdebate"]}')
        print(f'Post_predebate = "Post Intervention, before group discussion": {aggregated_mean_scores["post_predebate"]}')
        print(f'Post_postdebate = "Post Intervention, after group discussion": {aggregated_mean_scores["post_postdebate"]}')
        print()
        print("Mean Power Values Across all Simulations:")
        print(f"F-statistic: {aggregated_mean_t_stat}")
        print(f"P-value: {aggregated_mean_p_val}")
        print(f"Cohen's f: {aggregated_mean_cohens_f}")
        print(f"Power: {aggregated_mean_power}")
        print(f"Required Number of Participants for power 0.8 (Cohen's f): {aggregated_mean_n_required_power_0_8_f:.2f}")
        print()
        print("Cohen's d and Power Analysis for 'Pre_predebate' vs 'Post_predebate':")
        print(f"Cohen's d: {aggregated_mean_cohens_d_pre_post}")
        print(f"Power (Cohen's d): {aggregated_mean_power_cohens_d_pre_post}")
        print(f"T-statistic (Pre-Post): {aggregated_mean_t_stat_pre_post}")
        print(f"P-value (Pre-Post): {aggregated_mean_p_val_pre_post}")
        print(f"Power (Pre-Post): {aggregated_mean_power_pre_post}")
        print(f"Sample size needed for power of 0.8 (Cohen's d): {aggregated_mean_n_required_power_0_8_pre_post:.2f}")
        print()

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
