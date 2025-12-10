import matplotlib.pyplot as plt
import seaborn as sns

def visualize_ab(df, title):
    fig, ax = plt.subplots(1, 3, figsize=(16,5))

    df.groupby('ab_group')['has_claim'].mean().plot.bar(ax=ax[0])
    ax[0].set_title("Claim Frequency (%)")

    claims = df[df['totalclaims'] > 0]
    sns.boxplot(data=claims, x='ab_group', y='totalclaims', ax=ax[1])
    ax[1].set_title("Claim Severity")

    df['ab_group'].value_counts().plot.bar(ax=ax[2])
    ax[2].set_title("Sample Sizes")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
