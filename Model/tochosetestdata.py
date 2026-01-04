import pandas as pd

# Load dataset
df = pd.read_csv(
    r"C:\Users\Asus\OneDrive\Documents\DATASET\TrainingandTestingSets\UNSW_NB15_training-set.csv"
)

print(df["attack_cat"].value_counts())

# -----------------------------
# Sampling strategy
# -----------------------------
NORMAL_SAMPLES = 200
ATTACK_SAMPLES_PER_CLASS = 15

df_normal = df[df["attack_cat"] == "Normal"].sample(
    n=NORMAL_SAMPLES,
    random_state=42
)

df_attacks = (
    df[df["attack_cat"] != "Normal"]
    .groupby("attack_cat", group_keys=False)
    .apply(lambda x: x.sample(
        n=min(len(x), ATTACK_SAMPLES_PER_CLASS),
        random_state=42
    ))
)

# Combine
df_final = pd.concat([df_normal, df_attacks])

# Shuffle ONCE (important)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_final.to_csv("Testing.csv", index=False)

print("Saved Testing.csv with distribution:")
print(df_final["attack_cat"].value_counts())
