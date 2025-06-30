import pandas as pd
from IPython.display import display

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


def check_preprocessing(X_prepared):
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


def check_split(X_train, X_test, y_train, y_test, y):
    try:
        n_total = len(y)
        n_train_expected = int(0.8 * n_total)
        n_test_expected = n_total - n_train_expected

        # Shape checks
        if len(X_train) != n_train_expected or len(X_test) != n_test_expected:
            print("❌ Falsche Aufteilung der Daten.")
            return

        # Stratification check
        from collections import Counter
        def rel_freq(arr): return Counter(arr)  # z. B. {0: 102, 1: 23}
        train_freq = rel_freq(y_train)
        test_freq = rel_freq(y_test)
        orig_freq = rel_freq(y)

        if abs(train_freq[1]/len(y_train) - orig_freq[1]/len(y)) > 0.02:
            print("❌ Verteilung der Zielvariable stimmt nicht – eventuell fehlt `stratify=y`?")
            return

        print("✅ Aufteilung erfolgreich!")
    except Exception as e:
        print("❌ Fehler bei der Prüfung:", str(e))


def check_model_training(model, X_train, y_train):
    try:
        preds = model.predict(X_train)
        if len(preds) != len(y_train):
            print("❌ Modell scheint nicht korrekt trainiert.")
            return
        print("✅ Modell erfolgreich trainiert!")
    except Exception as e:
        print("❌ Fehler beim Modell:", str(e))
