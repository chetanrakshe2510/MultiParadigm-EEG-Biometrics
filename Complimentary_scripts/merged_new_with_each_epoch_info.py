import h5py
import os
import pandas as pd

# === Step 1: Setup paths ===
input_files = [
    'all_subject_eeg_features_S3_hierarchical.h5',
    'all_subject_eeg_features_S4_hierarchical.h5',
    'raw_all_subject_eeg_features_S5_hierarchical.h5',
    'all_subject_eeg_features_S6_hierarchical.h5',
    'all_subject_eeg_features_S7_hierarchical.h5',
    'all_subject_eeg_features_S8_hierarchical.h5',
    'all_subject_eeg_features_S9_hierarchical.h5',
    'all_subject_eeg_features_S10_hierarchical.h5',
    'all_subject_eeg_features_S11_hierarchical.h5',
    'all_sub_eyes_close_hierarchical.h5',
    'All_sub_EyesOpen_with_ICA.h5'
]

output_path = 'new_all_subjects_merged_new_full_epochs.h5'
mapping_csv  = 'epoch_mapping.csv'

# remove old outputs if present
for p in (output_path, mapping_csv):
    if os.path.exists(p):
        os.remove(p)

# === Step 2: Prepare provenance recording ===
# We'll accumulate one dict per epoch-copied
mapping_records = []

# epoch counters to give each (subject, run) its own sequence
epoch_counters = {}

def merge_and_reindex_epochs(src_group, dst_group, subject, run, src_file):
    key = (subject, run)
    if key not in epoch_counters:
        epoch_counters[key] = 0

    for orig_name in sorted(src_group.keys()):
        if isinstance(src_group[orig_name], h5py.Group) and orig_name.startswith("epoch"):
            new_idx = epoch_counters[key]
            new_name = f"epoch_{new_idx}"
            # copy the data
            dst_group.copy(src_group[orig_name], new_name)
            # record provenance
            mapping_records.append({
                "Subject": subject,
                "Run": run,
                "New_Epoch": new_name,
                "Source_File": os.path.basename(src_file),
                "Original_Epoch": orig_name
            })
            epoch_counters[key] += 1

# === Step 3: Perform merging + provenance capture ===
with h5py.File(output_path, 'w') as f_out:
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file_path}")
            continue

        with h5py.File(file_path, 'r') as f_in:
            for subject in f_in:
                grp_in = f_in[subject]
                grp_out = f_out.require_group(subject)

                for run in grp_in:
                    run_in  = grp_in[run]
                    run_out = grp_out.require_group(run)

                    merge_and_reindex_epochs(
                        run_in,
                        run_out,
                        subject,
                        run,
                        file_path
                    )

# === Step 4: Save provenance to CSV ===
df_map = pd.DataFrame(mapping_records)
df_map.to_csv(mapping_csv, index=False)
print(f"✅ Written mapping of {len(df_map)} epochs to '{mapping_csv}'")

# === (Optional) Step 5: Verify counts as before ===
def summarize_epochs(file_path):
    summary = []
    with h5py.File(file_path, 'r') as f:
        for subject in f:
            for run in f[subject]:
                count = sum(1 for name in f[subject][run] if name.startswith("epoch_"))
                summary.append({"Subject": subject, "Run": run, "Epoch_Count": count})
    return pd.DataFrame(summary)

summary_df = summarize_epochs(output_path)
print("\n🔍 Merged Epoch Summary (first few rows):")
print(summary_df.head())
