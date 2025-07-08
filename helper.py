import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(41)

# Modellparameter aus deiner letzten Anfrage
a = 30
b = 1.7
schwellenwerte = [1, 2, 5, 10, 20, 50, 100, 300]

def val(s):
    """Mittlere Zyklusanzahl bei Schwellenwert s."""
    return a * s**b

def symmetric_noise(spread):
    return np.random.normal(0, spread)

def generate_row(idx):
    beschichtung = np.random.choice(["Ja", "Nein"], p=[0.5, 0.5])
    zwischenschicht = np.random.choice(["Ja", "Nein"], p=[0.5, 0.5])
    normalkraft = round(np.random.choice([1., 2.5, 5., 2., 3., 4., 20.], p=[0.1, 0.5, 0.1, 0.2, 0.04, 0.03, 0.03]), 1)
    frequenz = round(np.random.choice([1.0, 1.5, 2.0], p=[0.3, 0.5, 0.2]), 1)
    hub = np.random.choice([100, 150, 200, 400], p=[0.6, 0.15, 0.2, 0.05])

    defekt_score = (
        (1 / normalkraft) * 1.5 +
        (hub / 100) * 0.5 +
        frequenz * 0.7 +
        (0 if beschichtung == "Ja" else 0.3) +
        (0 if zwischenschicht == "Ja" else 0.2)
    )

    # Frühversager-Faktor: deutlich kleinere mittlere Lebensdauer
    if defekt_score > 2.9:
        a_eff = 5 + symmetric_noise(2)
        # drastisch früheres Versagen
        b_eff = 1.3 + symmetric_noise(0.15)
        std_factor = 0.3
    else:
        a_eff = a + symmetric_noise(5)
        b_eff = b + symmetric_noise(0.2)
        std_factor = 0.4

    max_cycles = 4e5
    reached = []
    last_reached = True
    for s in schwellenwerte:
        if s == 300:
            p_reach = min(0.95, max(0.1, defekt_score / (2.8 + 0.02 * s))) * 1.5  # Erhöhte Wahrscheinlichkeit
        else:
            p_reach = min(0.95, max(0.1, defekt_score / (2.8 + 0.02 * s)))
        if last_reached and np.random.rand() < p_reach:
            mu = a_eff * s**b_eff
            std = mu * std_factor
            cycle_val = int(np.clip(np.random.normal(mu, std), 0, max_cycles))
            reached.append(cycle_val)
        else:
            reached.append(-1)
            last_reached = False

    row = {
        "Datei": f"Sample-{idx}.txt",
        "Beschichtung_Ag_Sn": beschichtung,
        "Zwischenschicht_Ni": zwischenschicht,
        "Normalkraft": normalkraft,
        "Frequenz": frequenz,
        "Bewegungshub": hub,
    }
    for s, z in zip(schwellenwerte, reached):
        row[f"Zyklus_bei_{s}_mOhm"] = z

    return row


# Beispiel: 5 Zeilen erzeugen
df = pd.DataFrame([generate_row(i) for i in range(250)])
df.head()

print("Count: ", df[df["Zyklus_bei_300_mOhm"] > -1].count())

# Datei speichern
csv_path = "Data/Schwellenwerte-Table 2.csv"
df.to_csv(csv_path, index=False)


# 1. Alle relevanten Spalten identifizieren
zyklus_cols = [col for col in df.columns if col.startswith("Zyklus_bei")]


überschreitungen = (df[zyklus_cols] != -1).sum()

# 3. Schwellenwerte extrahieren (z. B. aus 'Zyklus_bei_300_mOhm' → '300')
schwellenwerte = [int(col.replace("Zyklus_bei_", "").replace("_mOhm", "")) for col in überschreitungen.index]

# 5. Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=schwellenwerte, y=überschreitungen)
plt.title("Anzahl der Überschreitungen pro Schwellenwert")
plt.ylabel("Anzahl")
plt.xlabel("Widerstandsschwelle in mΩ")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Nur die Zyklus-Spalten extrahieren
df_visual = df[zyklus_cols].copy().replace(-1, np.nan)

# Plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_visual, orient="h", palette="Set3")
plt.title("Boxplot der Zyklenzahlen beim Überschreiten der Schwellen")
plt.xlabel("Zyklenanzahl")
plt.ylabel("Widerstandsschwelle")
plt.grid(axis="x")
plt.show()