# import pandas as pd
# import numpy as np
# import glob
# from collections import Counter
# import matplotlib.pyplot as plt

# def process_love_island_data(file_paths):
#     all_partners_counts = []
#     all_relationship_lengths = []
    
#     for file in file_paths:
#         # 1. Load and clean the table
#         df = pd.read_csv(file)
        
#         # Drop completely empty rows and rename the first column to 'Islander'
#         df = df.dropna(subset=[df.columns[0]]).reset_index(drop=True)
#         df = df.rename(columns={df.columns[0]: 'Islander'})
        
#         # Identify columns that represent recoupling days (ignoring 'Final' and Unnamed columns)
#         day_cols = [c for c in df.columns if 'Day' in c and c != 'Final']
        
#         for idx, row in df.iterrows():
#             # Get the chronological list of partners for the islander
#             partners = [p for p in row[day_cols] if pd.notna(p) and p != 'Not in Villa']
            
#             # --- Metric 1: Number of Unique Partners ---
#             unique_partners = set(partners)
#             all_partners_counts.append({
#                 'Islander': row['Islander'], 
#                 'Num_Partners': len(unique_partners),
#                 'Season_File': file
#             })
            
#             # --- Metric 2: Relationship Lengths ---
#             # Measured by how many consecutive checkpoints they stayed with the same person
#             if not partners:
#                 continue
                
#             current_partner = partners[0]
#             current_length = 1
            
#             for p in partners[1:]:
#                 if p == current_partner:
#                     # Survived another recoupling together!
#                     current_length += 1
#                 else:
#                     # Recoupled with someone else, record the length of the previous relationship
#                     all_relationship_lengths.append(current_length)
#                     current_partner = p
#                     current_length = 1
                    
#             # Append the length of their final relationship
#             all_relationship_lengths.append(current_length)
            
#     return pd.DataFrame(all_partners_counts), all_relationship_lengths

# # Example usage for your 12 tables:
# # Replace the path with the location of your files, e.g., 'data/season_*.csv'
# file_list = glob.glob('season_*_data.csv') 

# # Run extraction
# df_partners, relationship_lengths = process_love_island_data(file_list)

# # --- Display Results ---
# print("=== Number of Partners Distribution ===")
# partner_dist = df_partners['Num_Partners'].value_counts().sort_index()
# print(partner_dist)

# print("\n=== Relationship Length Distribution ===")
# # Length is in units of "consecutive coupling periods"
# length_dist = dict(sorted(Counter(relationship_lengths).items()))
# print(length_dist)

# # Optional: Plotting the distributions
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Plot Partner Distribution
# ax1.bar(partner_dist.index, partner_dist.values, color='skyblue', edgecolor='black')
# ax1.set_title('Number of Partners per Islander')
# ax1.set_xlabel('Total Unique Partners')
# ax1.set_ylabel('Frequency')
# ax1.set_xticks(partner_dist.index)

# # Plot Relationship Length Distribution
# ax2.bar(length_dist.keys(), length_dist.values(), color='lightcoral', edgecolor='black')
# ax2.set_title('Relationship Length Distribution')
# ax2.set_xlabel('Length (Consecutive Checkpoints)')
# ax2.set_ylabel('Frequency')
# ax2.set_xticks(list(length_dist.keys()))

# plt.tight_layout()
# plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

csv_files = [
    "season_5_data.csv",
    # "season_2_data.csv",
    # "season_3_data.csv",
]

INVALID_STRINGS = [
    "Not in Villa",
    "Dumped",
    "Withdrawn",
]

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def is_valid_partner(x):

    if pd.isna(x):
        return False

    x = str(x).strip()

    return not any(
        invalid.lower() in x.lower()
        for invalid in INVALID_STRINGS
    )


def find_coupling_columns(df):

    excluded = {
        "Contestant",
        "Unnamed: 0",
        "Notes"
    }

    return [
        c for c in df.columns
        if c not in excluded
    ]


# --------------------------------------------------
# COLLECT DATA ACROSS SEASONS
# --------------------------------------------------

partner_counts = []
relationship_lengths = []

for file in csv_files:

    season = file.replace(".csv", "")

    df = pd.read_csv(file)

    # Standardize contestant column
    if "Unnamed: 0" in df.columns:
        df = df.rename(
            columns={"Unnamed: 0": "Contestant"}
        )

    df = df[df["Contestant"].notna()]

    coupling_cols = find_coupling_columns(df)

    for _, row in df.iterrows():

        contestant = row["Contestant"]

        # ----------------------------------------
        # Partner count
        # ----------------------------------------

        partners = []

        for col in coupling_cols:

            partner = row[col]

            if is_valid_partner(partner):
                partners.append(str(partner))

        unique_partners = len(set(partners))

        partner_counts.append({
            "Season": season,
            "Contestant": contestant,
            "NumPartners": unique_partners
        })

        # ----------------------------------------
        # Relationship lengths
        # ----------------------------------------

        current_partner = None
        current_length = 0

        for col in coupling_cols:

            partner = row[col]

            if not is_valid_partner(partner):

                if current_partner is not None:
                    relationship_lengths.append(
                        current_length
                    )

                current_partner = None
                current_length = 0
                continue

            partner = str(partner)

            if partner == current_partner:

                current_length += 1

            else:

                if current_partner is not None:
                    relationship_lengths.append(
                        current_length
                    )

                current_partner = partner
                current_length = 1

        if current_partner is not None:
            relationship_lengths.append(
                current_length
            )

# --------------------------------------------------
# DATAFRAMES
# --------------------------------------------------

partner_counts = pd.DataFrame(partner_counts)

relationship_lengths = pd.DataFrame({
    "Length": relationship_lengths
})

# --------------------------------------------------
# PARTNER COUNT DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    partner_counts["NumPartners"],
    discrete=True
)

plt.xlabel("Number of Unique Partners")
plt.ylabel("Contestants")
plt.title("Partner Count Distribution")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# RELATIONSHIP LENGTH DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    relationship_lengths["Length"],
    discrete=True
)

plt.xlabel("Relationship Length (Coupling Periods)")
plt.ylabel("Count")
plt.title("Relationship Length Distribution")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

print("\nPARTNER COUNT")
print(partner_counts["NumPartners"].describe())

print("\nRELATIONSHIP LENGTH")
print(relationship_lengths["Length"].describe())