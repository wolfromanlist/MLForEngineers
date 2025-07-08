import pandas as pd
from IPython.display import display


def check_preprocessing_pipeline(df, X_processed, X_train, X_test, y_train, y_test):
    from numpy import allclose

    # === 1. Prüfung: Echte NaNs entfernt? ===
    dropped_idx = 25
    if dropped_idx in set(df.index):
        print("❌ Zeile mit echtem NaN wurde nicht entfernt (Index 25).")
        print("Tipp: Diese Zeile enthält eine echte Messlücke.")
        print("Solche Werte sollten nicht geschätzt oder ersetzt werden, da das Modell sonst verzerrt wird.")
        print("→ Verwende `dropna()` vor der Ersetzung der -1-Werte.")
        return

    # === 2. Prüfung: -1 korrekt durch np.nan ersetzt? ===
    if (df == -1).sum().sum() > 0:
        print("❌ Einige -1-Werte sind noch vorhanden. Diese sollten durch `np.nan` ersetzt werden.")
        return
    
    print("✅ Alle echten NaNs entfernt und -1 korrekt durch NaN ersetzt.")

    # === 3. Prüfung: Zielvariable 'target' korrekt erstellt? ===
    if "target" not in df.columns:
        print("❌ Zielvariable 'target' fehlt.")
        return
    expected_target = (df["Zyklus_bei_300_mOhm"] > 0).astype(int)
    if not df["target"].equals(expected_target):
        print("❌ Zielvariable 'target' nicht korrekt. Es gibt Diskrepanzen in der Kodierung.")
        n_wrong = (df["target"] != expected_target).sum()
        print(f"→ {n_wrong} fehlerhafte Zeilen.")
        return

    print("✅ Zielvariable korrekt erstellt.")

    # === 4. Prüfung: One-Hot-Encoding vorhanden und korrekt? ===
    dummy_cols = [col for col in X_processed.columns if "Beschichtung_Ag_Sn_Nein" in col or "Zwischenschicht_Ni_Nein" in col]
    if "Beschichtung_Ag_Sn_Ja" in X_processed.columns or "Zwischenschicht_Ni_Ja" in X_processed.columns:
        print("❌ Kategorische Variablen wurden nicht korrekt encodiert.")
        print("Verwende das keyword `drop_first=True` in `pd.get_dummies()`, um Multikollinearität zu vermeiden.")
        return
    if not dummy_cols:
        print("❌ Kategorische Variablen wurden nicht encodiert.")
        return

    print("✅ One-Hot-Encoding korrekt angewendet.")

    # === 5. Prüfung: Standardisierung der numerischen Features ===
    numerical_cols = ["Normalkraft", "Frequenz", "Bewegungshub"]
    if not all(col in X_processed.columns for col in numerical_cols):
        print("❌ Einige numerische Features fehlen:", [col for col in numerical_cols if col not in X_processed.columns])
        return

    means = X_processed[numerical_cols].mean()
    stds = X_processed[numerical_cols].std(ddof=0)
    print("\n Berechnete Mittelwerte und Standardabweichungen: \n")
    print("Mittelwerte: \n", means)
    print("Standardabweichungen: \n", stds, "\n")
    if not allclose(means, 0, atol=0.1):
        print("❌ Die numerischen Features sind nicht korrekt zentriert (Mittelwert ≠ 0).")
        print(means)
        return
    if not allclose(stds, 1, atol=0.1):
        print("❌ Die numerischen Features sind nicht korrekt skaliert (Std ≠ 1).")
        print(stds)
        return

    print("✅ Numerische Features korrekt standardisiert.")

    # === 6. Prüfung: Train-Test-Split ===

    try:
        n_total = len(X_train) + len(X_test)
        n_train_expected = int(0.8 * n_total)
        n_test_expected = n_total - n_train_expected

        # Shape checks (Abweichung von 1 erlauben)
        if abs(len(X_train) - n_train_expected) > 1 or abs(len(X_test) - n_test_expected) > 1:
            print("❌ Falsche Aufteilung der Daten.")
            # Ausgabe der tatsächlichen Aufteilung als Prozentanteil
            train_pct = len(X_train) / n_total * 100
            test_pct = len(X_test) / n_total * 100
            print("Erwartete Aufteilung: 80% Training, 20% Test")
            print("Tatsächliche Aufteilung:")
            print(f"Train: {len(X_train)} vs. Erwartet: {n_train_expected}")
            print(f"Test:  {len(X_test)} vs. Erwartet: {n_test_expected}")
            return
        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
            print("❌ Features und Zielgrößen haben unterschiedliche Länge.")
            return
    except Exception as e:
        print("❌ Fehler bei der Prüfung:", str(e))

    print("✅ Train-Test-Split korrekt!")

    print("\n Alle Preprocessing-Schritte erfolgreich durchgeführt!")

def check_model_training(model, X_train, y_train):
    import numpy as np
    tolerance = 0.1  # Toleranz für Koeffizientenvergleich
    reference_coef = [-0.50031861, 0.39139248, 1.16948728, 0.12410654, 0.11652089]

    try:
        # 1. Prüfung: Modell trainiert?
        preds = model.predict(X_train)
        if len(preds) != len(y_train):
            print("❌ Modell scheint nicht korrekt trainiert.")
            return
        
        # 2. Prüfung: Modell hat Koeffizienten?
        if not hasattr(model, "coef_"):
            print("❌ Modell enthält keine gelernten Koeffizienten (`model.coef_`).")
            return

        # 3. Optional: Vergleiche mit Referenzparametern
        if reference_coef is not None:
            learned_coef = model.coef_.flatten()
            reference_coef = np.array(reference_coef).flatten()

            if learned_coef.shape != reference_coef.shape:
                print("❌ Die Form der Koeffizienten stimmt nicht mit der Referenz überein.")
                return

            diff = np.abs(learned_coef - reference_coef)
            if np.any(diff > tolerance):
                print("⚠️ Modell wurde trainiert, aber die Koeffizienten weichen deutlich von der Referenz ab.")
                print("   → Mögliche Ursache: falsche Features, unstandardisierte Daten, fehlende Dropna etc.")
                print("   Max. Abweichung:", np.max(diff))
                return

        print("✅ Modell erfolgreich trainiert und Koeffizienten korrekt.")
    
    except Exception as e:
        print("❌ Fehler beim Modelltraining:", str(e))


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
