
title_block = """
# Data-Driven Donor Engagement in the Nonprofit Sector
# Unlocking Insights with Segmentation and Impact Alignment
#
# Author: Mahesh Pandey
# Project Type: Data Analytics | Nonprofit Sector | IIBA-aligned
"""
# 1. Executive Summary
# 2. Introduction: Why This Matters
# 3. About the Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# display settings for consistent plot style
sns.set(style='whitegrid')
plt.rcParams['figure.dpi'] = 100

#base patch
base_path = 'C:/GitHub/donor-data-analytics/Data/'

# Load datasets
donors = pd.read_csv(base_path + 'donors.csv')
donations = pd.read_csv(base_path + 'donations.csv')
donations_linked = pd.read_csv(base_path + 'donations_linked.csv')
engagements = pd.read_csv(base_path + 'engagement_history.csv')
campaigns = pd.read_csv(base_path + 'campaigns.csv')
impact = pd.read_csv(base_path + 'impact.csv')

entities = ['Donors', 'Donations', 'Campaigns', 'Engagements', 'Impact Records']
counts = [donors.shape[0], donations.shape[0], campaigns.shape[0], engagements.shape[0], impact.shape[0]]

# Plot
plt.figure(figsize=(8, 4))
plt.bar(entities, counts, color='skyblue')
plt.title('Dataset Overview: Entity Counts')
plt.xlabel('Count')
plt.tight_layout()
plt.show()

# 4. Using Data to Understand Engagement
# 4.1 Donor Behaviour & Recency Analysis
# Group by DonorID to get donation count
donation_counts = donations.groupby('DonorID')['DonationID'].count()

# Plot distribution
plt.figure(figsize=(8, 4))
sns.histplot(donation_counts, bins=20, kde=False, color='blue')
plt.title('Number of Donations per Donor')
plt.xlabel('Donation Count')
plt.ylabel('Number of Donors')
plt.tight_layout()
plt.show()

# Convert donation dates to datetime
donations['DonationDate'] = pd.to_datetime(donations['DonationDate'])

# Calculate recency per donor
last_donation = donations.groupby('DonorID')['DonationDate'].max()
recency_days = (pd.to_datetime('today') - last_donation).dt.days

# Plot recency
plt.figure(figsize=(8, 4))
sns.histplot(recency_days, bins=30, kde=False, color='Blue')
plt.title('Recency of Last Donation (in Days)')
plt.xlabel('Days Since Last Donation')
plt.ylabel('Number of Donors')
plt.tight_layout()
plt.show()

# 4.2 Donor Engagement Patterns & Segmentation Logic
# Convert donation date to datetime
donations['DonationDate'] = pd.to_datetime(donations['DonationDate'])

# Create donor-level summary
today = pd.to_datetime('today')
donor_summary = donations.groupby('DonorID').agg(
    FirstDonationDate=('DonationDate', 'min'),
    LastDonationDate=('DonationDate', 'max'),
    Frequency=('DonationID', 'count'),
    Monetary=('Amount', 'sum')
).reset_index()
donor_summary['Recency'] = (today - donor_summary['LastDonationDate']).dt.days

# Merge with full donor list
donor_full_summary = pd.merge(donors[['DonorID']], donor_summary, on='DonorID', how='left')
donor_full_summary['Recency'] = donor_full_summary['Recency'].fillna(np.inf)
donor_full_summary['Frequency'] = donor_full_summary['Frequency'].fillna(0)
donor_full_summary['Monetary'] = donor_full_summary['Monetary'].fillna(0)

# Assign segments
def assign_segment(row):
    r = row['Recency']
    f = row['Frequency']
    m = row['Monetary']

    if np.isinf(r) or pd.isna(r):
        return 'Never Donated'
    elif r <= 365 and f == 1:
        return 'New Donors'
    elif r <= 365 and f >= 4 and m >= 750:
        return 'Champions'
    elif r <= 1095 and f >= 4:
        return 'Loyal Donors'
    elif r <= 1095 and 2 <= f <= 4 and m >= 750:
        return 'High Value Potentials'
    elif r > 1095 and f >= 2 and m >= 250:
        return 'Lapsed but Valuable'
    elif 365 < r <= 1095 and f >= 2:
        return 'At Risk'
    elif r > 1095 and f == 1:
        return 'Lost or Inactive'
    elif f <= 2 and m < 250:
        return 'Low Frequency'
    else:
        return 'Misc Donors'

donor_full_summary['Segment'] = donor_full_summary.apply(assign_segment, axis=1)

# Return segment summary for confirmation
segment_summary = donor_full_summary['Segment'].value_counts().reset_index()
segment_summary.columns = ['Segment', 'Count']
segment_summary

# 📊 RFM Segmentation Logic (Applied)
# 4.3 Engagement History – Closing the Feedback Loop
engagements_outcomes = engagements.merge(donor_full_summary[['DonorID', 'Segment']], on='DonorID', how='left')

# Create a countplot showing outcomes across segments
plt.figure(figsize=(12, 6))
sns.countplot(data=engagements_outcomes, x='EngagementOutcome', hue='Segment', palette='husl')
plt.title('Engagement Outcomes by Donor Segment')
plt.xlabel('Engagement Outcome')
plt.ylabel('Number of Engagements')
plt.xticks(rotation=0)
plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Sort campaigns by TargetAmount

campaigns['Gap (%)'] = ((campaigns['ActualAmount'] - campaigns['TargetAmount']) / campaigns['TargetAmount']) * 100

campaigns_sorted = campaigns.sort_values('Gap (%)', ascending=True).reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=campaigns_sorted, x='Gap (%)', y='CampaignName', hue = 'CampaignName', palette='coolwarm')
plt.axvline(0, color='grey', linestyle='--')
plt.title('Campaign Fundraising Performance Gap (%)')
plt.xlabel('Performance Gap (%)')
plt.ylabel('Campaign')
plt.tight_layout()
plt.show()


# 4.4 Campaign Participation by Donor Segment
# Merge donations with donor segments from donor_full_summary
merged = donations.merge(donor_full_summary[['DonorID', 'Segment']], on='DonorID', how='left')

# Also bring in campaign names
merged = merged.merge(campaigns[['CampaignID', 'CampaignName']], on='CampaignID', how='left')

# Calculate total donations per segment per campaign
segment_campaign_totals = merged.groupby(['CampaignName', 'Segment'])['Amount'].sum().reset_index()

# Pivot to heatmap format
segment_pivot = segment_campaign_totals.pivot(index='Segment', columns='CampaignName', values='Amount').fillna(0)

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(segment_pivot, cmap='YlGnBu', annot=True, fmt='.0f', linewidths=0.5)
plt.title('Segment-wise Donation Totals ($) by Campaign')
plt.xlabel('Campaign')
plt.ylabel('Donor Segment')
plt.tight_layout()
plt.show()



# 4.5 – Campaign Performance & Impact Alignment
# Ensure the value and cost columns are numeric
impact['Value'] = pd.to_numeric(impact['Value'], errors='coerce')
impact['Cost'] = pd.to_numeric(impact['Cost'], errors='coerce')

# Compute total impact cost per entry
impact['ValueDelivered'] = impact['Value'] * impact['Cost']
impact_total = impact.groupby('CampaignID')['ValueDelivered'].sum().reset_index(name='ImpactDelivered_$')
merged_campaigns = pd.merge(campaigns, impact_total, on='CampaignID', how='left')
merged_campaigns['ImpactEfficiency'] = merged_campaigns['ImpactDelivered_$'] / merged_campaigns['ActualAmount']

# Sort by impact efficiency
merged_campaigns = merged_campaigns.sort_values('CampaignName')

plt.figure(figsize=(10, 6))
sns.barplot(data=merged_campaigns, y='CampaignName', x='ImpactEfficiency', color='seagreen')
plt.title('Dollar Value of Impact Delivered per Dollar Raised')
plt.xlabel('Impact Delivered per $ Raised')
plt.ylabel('Campaign')
plt.tight_layout()
plt.show()


# 5. RFM Segmentation & Donor Profiling
# 5.1 Segment Definitions & Rules
# 5.2 Donor Segment Distribution
plt.figure(figsize=(9, 5))
sns.barplot(data=segment_summary, x='Count', y='Segment', hue = 'Segment', palette='bright')
plt.title('Donor Distribution by Segment')
plt.xlabel('Number of Donors')
plt.ylabel('Donor Segment')
plt.tight_layout()
plt.show()


# 5.3 Segment-Level Performance Analysis
# Merge segment info into donations
donations_with_segments = pd.merge(
    donations,
    donor_full_summary[['DonorID', 'Segment']],
    on='DonorID',
    how='left'
)

# Group by segment to get average donation
avg_donation = donations_with_segments.groupby('Segment')['Amount'].mean().reset_index()
avg_donation = avg_donation.sort_values('Amount', ascending=False)


# Plot
plt.figure(figsize=(10, 5))
sns.barplot(data=avg_donation, x='Amount', y='Segment', hue = 'Segment', palette='bright')

# Fixing axis and labels
plt.xlim(0, avg_donation['Amount'].max() + 50)
plt.title('Average Donation Amount by Donor Segment')
plt.xlabel('Average Donation ($)')
plt.ylabel('Donor Segment')

# Reset index to ensure clean annotation
avg_donation = avg_donation.reset_index(drop=True)

# Annotate with proper positioning
for i, row in avg_donation.iterrows():
    plt.text(row['Amount'] + 5, i, f"${int(row['Amount'])}", va='center', fontsize=10)

plt.tight_layout()
plt.show()


# Step 1: Filter engaged outcomes only (if needed)
engaged_df = engagements[engagements['EngagementOutcome'] == 'Engaged']

# Step 2: Merge with segments
engaged_df = engaged_df.merge(donor_full_summary[['DonorID', 'Segment']], on='DonorID', how='left')

# Step 3: Count engagements per segment
engagement_counts = engaged_df.groupby('Segment').size().reset_index(name='EngagedInteractions')

# Step 4: Total donors per segment
donor_counts = donor_full_summary['Segment'].value_counts().reset_index()
donor_counts.columns = ['Segment', 'TotalDonors']

# Step 5: Merge and calculate average engagements per donor
engagement_avg = pd.merge(engagement_counts, donor_counts, on='Segment', how='left')
engagement_avg['AvgEngagementsPerDonor'] = engagement_avg['EngagedInteractions'] / engagement_avg['TotalDonors']

# Step 6: Plot it
engagement_avg = engagement_avg.sort_values('AvgEngagementsPerDonor', ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=engagement_avg, x='AvgEngagementsPerDonor', y='Segment', hue = 'Segment', palette='crest')
plt.xlabel('Average Engagements per Donor')
plt.ylabel('Donor Segment')
plt.title('Figure 10. Average Engagements per Donor by Segment')
plt.tight_layout()
plt.show()


# 5.4 Missed Opportunities
# Merge segment info into donations
donations_with_segments = pd.merge(
    donations,
    donor_full_summary[['DonorID', 'Segment']],
    on='DonorID',
    how='left'
)

# Aggregate: total donation value and donor count per segment
segment_performance = donations_with_segments.groupby('Segment').agg({
    'Amount': 'sum',
    'DonorID': pd.Series.nunique
}).reset_index()

segment_performance.columns = ['Segment', 'TotalDonated', 'DonorCount']

# Sort by donation volume
segment_performance = segment_performance.sort_values('TotalDonated', ascending=False)


plt.figure(figsize=(10, 6))
sns.scatterplot(data=segment_performance, x='DonorCount', y='TotalDonated', hue='Segment', s=200, palette='tab10')

# Annotate each point
for i in range(segment_performance.shape[0]):
    plt.text(
        x=segment_performance['DonorCount'].iloc[i] + 5,
        y=segment_performance['TotalDonated'].iloc[i],
        s=segment_performance['Segment'].iloc[i],
        fontdict=dict(size=9)
    )

plt.title('Figure 13. Segment Performance: Donor Volume vs Total Donations')
plt.xlabel('Number of Donors')
plt.ylabel('Total Donations ($)')
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))  #$Format added
plt.tight_layout()
plt.show()

# 6. Recommendations
# 7. Conclusion
# 8. References








