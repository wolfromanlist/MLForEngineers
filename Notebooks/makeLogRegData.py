import numpy as np
import matplotlib.pyplot as plt

# Zufällige, aber reproduzierbare Daten
np.random.seed(42)

# Normalkraft: Werte zwischen 20 und 80 N
n_samples = 150
force = np.random.uniform(20, 80, n_samples)

# Kontaktwiderstand nimmt exponentiell mit der Kraft ab + etwas Rauschen
true_resistance = 500 * np.exp(-0.01 * force)
noise = np.random.normal(0, 20, n_samples)
resistance = true_resistance + noise

# Klassifikation: alles mit Widerstand > 300 mOhm ist defekt
is_defective = resistance > 300

# Erstellen des DataFrames
import pandas as pd
data = pd.DataFrame({
    'Normalkraft': force,
    'Widerstand': resistance
})

# speichern der Daten
data.to_csv('classification_data_resistance_force.csv', index=False)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(force[~is_defective], resistance[~is_defective],
            c='green', marker='s', label='nicht defekt')
plt.scatter(force[is_defective], resistance[is_defective],
            c='orange', marker='^', label='defekt')
plt.axhline(300, color='gray', linestyle='--', linewidth=1, label='Grenze: 300 mΩ')

plt.xlabel("Kontaktnormalkraft [N]")
plt.ylabel("Kontaktwiderstand [mΩ]")
plt.title("Beispielhafte Steckverbindungen: Defektklassifikation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
