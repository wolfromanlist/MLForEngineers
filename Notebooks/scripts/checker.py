import pandas as pd
from IPython.display import display

def check_handling_missing_values(df_after):
    dropped_idx = 25
    still_present = dropped_idx in set(df_after.index)

    # 1. Wurde Zeile mit echtem NaN entfernt?
    if still_present:
        print(f"❌ Zeile {dropped_idx} mit echten fehlenden Werten wurde nicht entfernt.")
        print("💡 Tipp: Diese Zeile enthält echte Messlücken (z. B. bei physikalischen Eigenschaften).")
        print("    Solche Werte sollten nicht geschätzt oder ersetzt werden, da das Modell sonst verzerrt wird.")
        print("    → Verwende `dropna()` vor der Ersetzung der -1-Werte.")
        return

    # 2. Wurden -1 durch NaN ersetzt?
    if (df_after == -1).sum().sum() > 0:
        print("❌ Einige -1 sind noch im DataFrame – bitte durch NaN ersetzen.")
        return

    print("✅ Alle echten NaNs entfernt und -1 korrekt durch NaN ersetzt. \n Info: \n")
    print(df_after.info())

def check_target_column(df):
    if "target" not in df.columns:
        print("❌ Fehler: Die Spalte 'target' wurde nicht erstellt.")
        return

    expected = (df["Zyklus_300_mOhm"] > 0).astype(int)
    if not df["target"].equals(expected):
        print(f"❌ 'target' ist nicht korrekt. {sum(df['target'] != expected)} Fehlerhafte Zeilen.")
    else:
        print("✅ Zielvariable korrekt erstellt: \n", df[['target']])


def check_preprocessing(X_prepared):
    # Erwartete Spaltennamen nach One-Hot-Encoding
    must_include = ["Normalkraft", "Frequenz", "Bewegungshub", "Zyklus_1_mOhm", "Zyklus_2_mOhm", "Zyklus_5_mOhm", "Zyklus_10_mOhm", "Zyklus_20_mOhm", "Zyklus_50_mOhm", "Zyklus_100_mOhm", "Zyklus_300_mOhm"]
    dummy_cols = [col for col in X_prepared.columns if "Beschichtung_Ag_Sn" in col or "Zwischenschicht_Ni" in col]

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


def check_metrics(y_test, y_pred, y_prob):
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        if not set(y_pred).issubset({0, 1}):
            print("❌ y_pred enthält keine gültigen Klassen (0/1).")
            return
        if not (0 <= y_prob.min() <= 1 and 0 <= y_prob.max() <= 1):
            print("❌ y_prob scheint keine Wahrscheinlichkeiten zu enthalten.")
            return

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"✅ Accuracy: {acc:.2f}")
        print(f"✅ Precision: {prec:.2f}")
        print(f"✅ Recall: {rec:.2f}")
        print(f"✅ F1-Score: {f1:.2f}")
        print(f"✅ ROC AUC: {auc:.2f}")
    except Exception as e:
        print("❌ Fehler beim Berechnen der Metriken:", str(e))


def check_coefficients(model, X_prepared):
    try:
        import pandas as pd
        if not hasattr(model, "coef_"):
            print("❌ Modell enthält keine Koeffizienten.")
            return
        coefs = model.coef_[0]
        if len(coefs) != X_prepared.shape[1]:
            print("❌ Die Anzahl der Koeffizienten stimmt nicht mit den Features überein.")
            return
        print("✅ Modell-Koeffizienten korrekt extrahiert.")
    except Exception as e:
        print("❌ Fehler bei der Koeffizienten-Analyse:", str(e))
