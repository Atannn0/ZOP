import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_rel, wilcoxon, shapiro, ttest_1samp
import seaborn as sns

# Load the data
file_path = 'Důvěryhodnost umělé inteligence (Odpovědi)-4.xlsx'
data = pd.ExcelFile(file_path)
responses = data.parse('Odpovědi formuláře 1')

# Preprocess the data
responses.columns = [
    'timestamp', 'age', 'gender', 'education', 'technical_skills',
    'ai_feelings', 'trust_when_explained', 'importance_of_transparency',
    'trust_step_by_step', 'trust_daily_decisions', 'trust_health_finance',
    'ai_without_bias', 'ai_protected', 'trust_org_types', 'ai_risks',
    'increase_trust_ai', 'trust_scenario'
]

# Popis respondentů
respondents_summary = responses[['age', 'gender', 'education', 'technical_skills']].describe(include='all')
print("### Popis respondentů ###")
print(respondents_summary)

# Set dark theme for plots
plt.style.use('dark_background')

# H1: Emoce a demografické faktory
positive_emotions = ['Optimistický/á', 'Bezstarostný/á', 'Nadšený/á']
negative_emotions = ['Znepokojený/á', 'Bojácný/á', 'Pobouřený/á']

responses['ai_emotion_extended'] = responses['ai_feelings'].map(
    lambda x: 'Pozitivní' if x in positive_emotions else ('Negativní' if x in negative_emotions else 'Neutral')
)

demographic_vars = {
    'age': 'Věk',
    'technical_skills': 'Technická zdatnost',
    'education': 'Vzdělání'
}

# Define custom colors
primary_color = "#6EA5B8"  # Light blue
secondary_color = "#F7C04A"  # Gold
neutral_color = "#474747"  # Gray

print("\n### H1: Emoce a demografické faktory ###")
for demo_var, label in demographic_vars.items():
    contingency_table = responses.groupby([demo_var, 'ai_emotion_extended']).size().unstack(fill_value=0)
    ax = contingency_table.plot(
        kind='bar', stacked=True, figsize=(12, 6),
        color=[primary_color, secondary_color, neutral_color], alpha=0.8
    )
    plt.title(f'Distribuce emocí vůči AI podle: "{label}"', fontsize=16)
    plt.xlabel(label, fontsize=14)
    plt.ylabel('Počet respondentů', fontsize=14)
    plt.legend(title='Kategorie emocí', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add values above bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='center', fontsize=10, color="white")

    plt.show()

    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    print(f'Chi-Square Test for {label}: Chi2-statistic = {chi2_stat:.2e}, p-value = {p_value:.2e}')
    if p_value < 0.05:
        print(f'Výsledek: Statisticky významný vztah mezi emocemi a {label}. Hypotéza H1 potvrzena pro {label}.\n')
    else:
        print(f'Výsledek: Statisticky nevýznamný vztah mezi emocemi a {label}. Hypotéza H1 není potvrzena pro {label}.\n')

# H2: Transparentnost a důvěra
response_mapping = {
    'Určitě ano': 5,
    'Spíše ano': 4,
    'Nerozhodnut/a': 3,
    'Spíše ne': 2,
    'Určitě ne': 1
}

responses['trust_when_explained_numeric'] = responses['trust_when_explained'].map(response_mapping)
responses['importance_of_transparency_numeric'] = responses['importance_of_transparency'].map(response_mapping)
responses['trust_step_by_step_numeric'] = responses['trust_step_by_step'].map(response_mapping)

responses['transparency_trust_score'] = responses[
    ['trust_when_explained_numeric', 'importance_of_transparency_numeric', 'trust_step_by_step_numeric']
].mean(axis=1)

print("\n### H2: Transparentnost a důvěra ###")
# Vizualizace distribuce dat
plt.hist(responses['transparency_trust_score'].dropna(), bins=10, color=primary_color, alpha=0.8, edgecolor='white')
plt.title('Distribuce skóre transparentnosti')
plt.xlabel('Průměrné skóre transparentnosti (1-5)')
plt.ylabel('Počet respondentů')
plt.tight_layout()
plt.show()

# Test normality
normality_stat, normality_pval = shapiro(responses['transparency_trust_score'].dropna())
print(f'Shapiro-Wilk Test: Statistik = {normality_stat:.4e}, p-value = {normality_pval:.2e}')

if normality_pval >= 0.05:
    print("Data odpovídají normálnímu rozdělení.")
    t_stat, p_value = ttest_1samp(responses['transparency_trust_score'].dropna(), 3)
    test_used = "T-test proti neutrální hodnotě (3)"
else:
    print("Data neodpovídají normálnímu rozdělení.")
    t_stat, p_value = wilcoxon(responses['transparency_trust_score'].dropna() - 3)
    test_used = "Wilcoxonův test proti neutrální hodnotě (3)"

print(f'{test_used}: Statistik = {t_stat:.4e}, p-value = {p_value:.2e}')
if p_value < 0.05:
    print("Výsledek: Transparentnost má statisticky významný vliv na důvěru. Hypotéza H2 potvrzena.\n")
else:
    print("Výsledek: Transparentnost nemá statisticky významný vliv na důvěru. Hypotéza H2 není potvrzena.\n")

# H3: Typ rozhodnutí a důvěra
print("\n### H3: Typ rozhodnutí a důvěra ###")

responses['trust_daily_decisions_numeric'] = responses['trust_daily_decisions'].map({
    'Velmi důvěřuji': 5,
    'Spíše důvěřuji': 4,
    'Nerozhodnut/a': 3,
    'Spíše nedůvěřuji': 2,
    'Vůbec nedůvěřuji': 1
})
responses['trust_health_finance_numeric'] = responses['trust_health_finance'].map({
    'Velmi důvěřuji': 5,
    'Spíše důvěřuji': 4,
    'Nerozhodnut/a': 3,
    'Spíše nedůvěřuji': 2,
    'Vůbec nedůvěřuji': 1
})

h3_data = responses[['trust_daily_decisions_numeric', 'trust_health_finance_numeric']].dropna()

daily_mean = h3_data['trust_daily_decisions_numeric'].mean()
health_finance_mean = h3_data['trust_health_finance_numeric'].mean()

# Vizualizace rozdílů v důvěře
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(
    ['Každodenní rozhodování', 'Zdraví a finance'], 
    [daily_mean, health_finance_mean], 
    color=[primary_color, secondary_color], alpha=0.8, edgecolor='white'
)

# Vizualizace rozdílů v důvěře
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(
    ['Každodenní rozhodování', 'Zdraví a finance'], 
    [daily_mean, health_finance_mean], 
    color=[primary_color, secondary_color], alpha=0.8, edgecolor='white'
)

# Add mean values as text on the bars
for bar, mean_value in zip(bars, [daily_mean, health_finance_mean]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,  # Center the text horizontally
        bar.get_height() / 2,  # Place the text vertically in the middle of the bar
        f'{mean_value:.2f}',  # Format to 2 decimal places
        ha='center', va='center', fontsize=12, color='white'
    )

plt.title('Průměrná důvěra v AI podle typu rozhodnutí', fontsize=16)
plt.ylabel('Průměrná důvěra (1-5)', fontsize=14)
plt.tight_layout()
plt.show()

# Distribuce dat jako hustotní grafy
plt.figure(figsize=(10, 6))
sns.kdeplot(h3_data['trust_daily_decisions_numeric'], fill=True, color="blue", label="Každodenní rozhodování", alpha=0.5)
sns.kdeplot(h3_data['trust_health_finance_numeric'], fill=True, color="orange", label="Zdraví a finance", alpha=0.5)
plt.title('Distribuce důvěry v AI podle typu rozhodnutí')
plt.xlabel('Důvěra (1-5)')
plt.ylabel('Hustota')
plt.legend()
plt.tight_layout()
plt.show()

# Statistický test
daily_normality = shapiro(h3_data['trust_daily_decisions_numeric'])
finance_normality = shapiro(h3_data['trust_health_finance_numeric'])

if daily_normality.pvalue >= 0.05 and finance_normality.pvalue >= 0.05:
    t_stat, p_value = ttest_rel(h3_data['trust_daily_decisions_numeric'], h3_data['trust_health_finance_numeric'])
    test_used = "T-test (párový)"
else:
    t_stat, p_value = wilcoxon(h3_data['trust_daily_decisions_numeric'], h3_data['trust_health_finance_numeric'])
    test_used = "Wilcoxonův test"

print(f'{test_used}: Statistik = {t_stat:.4e}, p-value = {p_value:.2e}')
if p_value < 0.05:
    print("Výsledek: Statisticky významný rozdíl v důvěře mezi typy rozhodování. Hypotéza H3 potvrzena.\n")
else:
    print("Výsledek: Statisticky nevýznamný rozdíl v důvěře mezi typy rozhodování. Hypotéza H3 není potvrzena.\n")

print(f'Průměrná důvěra - Každodenní rozhodování: {daily_mean:.2f}')
print(f'Průměrná důvěra - Zdraví a finance: {health_finance_mean:.2f}')
