import pandas as pd
from IPython.display import display

def check_overview(df):
    if not hasattr(df, "shape") or not hasattr(df, "columns"):
        print("❌ Das scheint kein gültiger DataFrame zu sein.")
        return
    print("✅ Form:", df.shape)
    print("✅ Spalten:", list(df.columns[:5]), "...")
    print("✅ Vorschau:")
    display(df.head())


def check_missing_and_types(df):
    na_counts = df.isna().sum()
    if not isinstance(na_counts, pd.Series):
        print("❌ Keine gültige Ausgabe für fehlende Werte.")
        return
    print("✅ Fehlende Werte (Ausschnitt):")
    display(na_counts[na_counts > 0].sort_values(ascending=False).head())

    print("\n✅ Datentypen:")
    display(df.dtypes.head())


def check_target_column(df):
    if "target" not in df.columns:
        print("❌ Fehler: Die Spalte 'target' wurde nicht erstellt.")
        return

    expected = (df["Zyklus_300_mOhm"].fillna(0) > 0).astype(int)
    if not df["target"].equals(expected):
        print(f"❌ 'target' ist nicht korrekt. {sum(df['target'] != expected)} Fehlerhafte Zeilen.")
    else:
        print("✅ Zielvariable korrekt erstellt!")


def check_preprocessing(df, X_prepared):
    # Erwartete Spaltennamen nach One-Hot-Encoding (abhängig von Daten!)
    must_include = ["Kontaktkraft_N", "Frequenz_Hz", "Hub_mm", "Steckzyklen"]
    dummy_cols = [col for col in X_prepared.columns if "Beschichtung_" in col or "Zwischenschicht_" in col]

    missing = [col for col in must_include if col not in X_prepared.columns]
    if missing:
        print("❌ Fehlende numerische Features:", missing)
        return

    if not dummy_cols:
        print("❌ Es wurden keine kategorischen Variablen encodiert.")
        return

    try:
        from numpy import allclose
        means = X_prepared[must_include].mean().abs()
        stds = X_prepared[must_include].std(ddof=0)
        if not allclose(means, 0, atol=1e-1):
            print("❌ Die numerischen Features sind nicht korrekt zentriert (Mittelwert ≠ 0).")
        elif not allclose(stds, 1, atol=1e-1):
            print("❌ Die numerischen Features sind nicht korrekt skaliert (Std ≠ 1).")
        else:
            print("✅ One-Hot-Encoding & Standardisierung korrekt!")
    except Exception as e:
        print("❌ Fehler bei der Überprüfung:", str(e))
