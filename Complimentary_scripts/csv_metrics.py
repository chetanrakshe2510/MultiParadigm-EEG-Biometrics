import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
input_dir   = r"G:\Publication and confernece\Workj_2\Results\Task_performance\Maha"
output_dir  = os.path.join(input_dir, "metrics_outputs")
os.makedirs(output_dir, exist_ok=True)

file_patterns = ["*.csv", "*.xlsx"]
combined_results = []
# ────────────────────────────────────────────────────────────────────────────────

for pattern in file_patterns:
    for file_path in glob.glob(os.path.join(input_dir, pattern)):
        name = os.path.splitext(os.path.basename(file_path))[0]
        print(f"→ Processing {name}")

        #  Load the file (CSV or Excel)
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        if not {'True_Label','Predicted_Label'}.issubset(df.columns):
            print(f"   ⚠ Skipped (missing True/Predicted columns)")
            continue

        y_true = df['True_Label']
        y_pred = df['Predicted_Label']

        #  Overall & avg precision/recall/F1
        acc    = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics = {
            'File': name,
            'Accuracy': acc,
            'Macro Precision': report['macro avg']['precision'],
            'Macro Recall':    report['macro avg']['recall'],
            'Macro F1 Score':  report['macro avg']['f1-score'],
            'Weighted Precision': report['weighted avg']['precision'],
            'Weighted Recall':    report['weighted avg']['recall'],
            'Weighted F1 Score':  report['weighted avg']['f1-score'],
        }

        #  FAR/FRR/EER per-class, then macro-EER
        per_cls = {}
        for cls in y_true.unique():
            bt = (y_true == cls).astype(int)
            bp = (y_pred == cls).astype(int)
            tn, fp, fn, tp = confusion_matrix(bt, bp).ravel()
            far = fp/(fp+tn) if (fp+tn) else 0
            frr = fn/(fn+tp) if (fn+tp) else 0
            per_cls[cls] = (far+frr)/2
        macro_eer = pd.Series(per_cls).mean()
        metrics['Macro EER'] = macro_eer

        combined_results.append(metrics)

#  Build combined DataFrame and save
combined_df = pd.DataFrame(combined_results).set_index('File')
out_csv = os.path.join(output_dir, "Combined_Multiclass_Performance_Metrics.csv")
combined_df.to_csv(out_csv)
print(f"\n✅ Combined metrics written → {out_csv}")
