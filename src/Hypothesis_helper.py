import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, ks_2samp

def calculate_claim_frequency(group_data):
    """Calculate proportion of policies with at least one claim"""
    return (group_data['has_claim'].sum() / len(group_data)) * 100

def calculate_claim_severity(group_data):
    """Calculate average claim amount for policies with claims > 0"""
    claims_with_amount = group_data[group_data['totalclaims'] > 0]
    if len(claims_with_amount) == 0:
        return 0
    return claims_with_amount['totalclaims'].mean()

def calculate_margin(group_data):
    """Calculate average margin (profit)"""
    return group_data['margin'].mean()

def print_test_result(hypothesis_num, null_hypothesis, test_name, statistic, 
                      p_value, alpha=0.05, effect_details=None):
    """Print formatted hypothesis test results"""
    print("\n" + "="*80)
    print(f"HYPOTHESIS {hypothesis_num}: {null_hypothesis}")
    print("="*80)
    print(f"Test Used: {test_name}")
    print(f"Test Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significance Level (α): {alpha}")
    
    if p_value < alpha:
        print(f"\n✗ REJECT NULL HYPOTHESIS (p < {alpha})")
        print("   → Significant differences detected!")
    else:
        print(f"\n✓ FAIL TO REJECT NULL HYPOTHESIS (p >= {alpha})")
        print("   → No significant differences detected")
    
    if effect_details:
        print(f"\nEffect Details:")
        print(effect_details)
    print("="*80)
def plot_group_comparison(data, group_col, metric_col, title, ylabel, 
                          top_n=None, figsize=(12, 6)):
    """Create visualization for group comparisons"""
    if top_n:
        top_groups = data.groupby(group_col)[metric_col].mean().sort_values(ascending=False).head(top_n).index
        plot_data = data[data[group_col].isin(top_groups)]
    else:
        plot_data = data
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot
    plot_data.boxplot(column=metric_col, by=group_col, ax=axes[0])
    axes[0].set_title(f'{title} - Distribution')
    axes[0].set_xlabel(group_col.capitalize())
    axes[0].set_ylabel(ylabel)
    axes[0].tick_params(axis='x', rotation=45)
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot of means
    means = plot_data.groupby(group_col)[metric_col].mean().sort_values(ascending=False)
    means.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title(f'{title} - Mean by Group')
    axes[1].set_xlabel(group_col.capitalize())
    axes[1].set_ylabel(f'Mean {ylabel}')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
def check_group_equivalence(group_a, group_b, features_to_check, alpha=0.05):
    """
    Verify that Group A and Group B are statistically equivalent on features
    other than the one being tested (to control for confounding variables)
    """
    print("\n" + "-"*80)
    print("CHECKING GROUP EQUIVALENCE (Controlling for Confounders)")
    print("-"*80)
    
    equivalence_results = []
    
    for feature in features_to_check:
        if feature not in group_a.columns or feature not in group_b.columns:
            continue
            
        # Skip if feature has too many missing values
        if group_a[feature].isna().sum() > len(group_a) * 0.5:
            continue
            
        # For numerical features
        if group_a[feature].dtype in ['float64', 'int64']:
            # Use t-test for numerical variables
            a_vals = group_a[feature].dropna()
            b_vals = group_b[feature].dropna()
            
            if len(a_vals) > 0 and len(b_vals) > 0:
                stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
                is_equivalent = "✓ EQUIVALENT" if p_val >= alpha else "✗ DIFFERENT"
                equivalence_results.append({
                    'Feature': feature,
                    'Test': 'T-Test',
                    'P-Value': p_val,
                    'Status': is_equivalent,
                    'Group_A_Mean': a_vals.mean(),
                    'Group_B_Mean': b_vals.mean()
                })
        
        # For categorical features
        elif group_a[feature].dtype in ['object', 'category', 'bool']:
            try:
                contingency = pd.crosstab(
                    pd.concat([group_a[feature], group_b[feature]]),
                    pd.Series(['A']*len(group_a) + ['B']*len(group_b))
                )
                if contingency.size > 1:
                    chi2, p_val, dof, expected = chi2_contingency(contingency)
                    is_equivalent = "✓ EQUIVALENT" if p_val >= alpha else "✗ DIFFERENT"
                    equivalence_results.append({
                        'Feature': feature,
                        'Test': 'Chi-Squared',
                        'P-Value': p_val,
                        'Status': is_equivalent,
                        'Group_A_Mean': '-',
                        'Group_B_Mean': '-'
                    })
            except:
                pass
    
    results_df = pd.DataFrame(equivalence_results)
    
    if len(results_df) > 0:
        print(results_df.to_string(index=False))
        
        # Count how many features are NOT equivalent
        non_equivalent = results_df[results_df['Status'].str.contains('DIFFERENT')]
        if len(non_equivalent) > 0:
            print(f"\n⚠ WARNING: {len(non_equivalent)} feature(s) show significant differences between groups!")
            print("   These may be confounding variables affecting the test:")
            print(non_equivalent[['Feature', 'P-Value']].to_string(index=False))
        else:
            print("\n✓ Groups are statistically equivalent on checked features")
    else:
        print("No features available for equivalence checking")
    
    print("-"*80)
    return results_df

def perform_ab_test(group_a, group_b, metric, test_type='ttest', group_a_name='Group A', group_b_name='Group B'):
    """
    Perform A/B test comparing a specific metric between two groups
    """
    a_vals = group_a[metric].dropna()
    b_vals = group_b[metric].dropna()
    
    if test_type == 'ttest':
        # T-test for continuous metrics
        stat, p_val = ttest_ind(a_vals, b_vals, equal_var=False)
        test_name = "Welch's T-Test"
    elif test_type == 'mannwhitney':
        # Mann-Whitney U test (non-parametric alternative)
        stat, p_val = mannwhitneyu(a_vals, b_vals, alternative='two-sided')
        test_name = "Mann-Whitney U Test"
    elif test_type == 'chi2':
        # Chi-squared test for binary/categorical
        contingency = pd.crosstab(
            pd.concat([group_a[metric], group_b[metric]]),
            pd.Series(['A']*len(group_a) + ['B']*len(group_b))
        )
        stat, p_val, dof, expected = chi2_contingency(contingency)
        test_name = "Chi-Squared Test"
    
    # Calculate effect size
    a_mean = a_vals.mean()
    b_mean = b_vals.mean()
    effect_size = b_mean - a_mean
    
    # Calculate relative difference
    if a_mean != 0:
        relative_diff = (effect_size / a_mean) * 100
    else:
        relative_diff = 0
    
    return {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_val,
        'group_a_mean': a_mean,
        'group_b_mean': b_mean,
        'effect_size': effect_size,
        'relative_diff_pct': relative_diff,
        'group_a_name': group_a_name,
        'group_b_name': group_b_name,
        'group_a_n': len(a_vals),
        'group_b_n': len(b_vals)
    }

def print_ab_test_result(hypothesis_num, null_hypothesis, result, alpha=0.05):
    """Print formatted A/B test results"""
    print("\n" + "="*80)
    print(f"HYPOTHESIS {hypothesis_num}: {null_hypothesis}")
    print("="*80)
    print(f"Group A (Control): {result['group_a_name']} (n={result['group_a_n']:,})")
    print(f"Group B (Test):    {result['group_b_name']} (n={result['group_b_n']:,})")
    print(f"\nTest Used: {result['test_name']}")
    print(f"Test Statistic: {result['statistic']:.4f}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"Significance Level (α): {alpha}")
    
    print(f"\n--- METRICS ---")
    print(f"Group A Mean: {result['group_a_mean']:.2f}")
    print(f"Group B Mean: {result['group_b_mean']:.2f}")
    print(f"Absolute Difference: {result['effect_size']:.2f}")
    print(f"Relative Difference: {result['relative_diff_pct']:.2f}%")
    
    if result['p_value'] < alpha:
        print(f"\n✗ REJECT NULL HYPOTHESIS (p < {alpha})")
        print("   → Significant differences detected between groups!")
    else:
        print(f"\n✓ FAIL TO REJECT NULL HYPOTHESIS (p >= {alpha})")
        print("   → No significant differences detected")
    
    print("="*80)