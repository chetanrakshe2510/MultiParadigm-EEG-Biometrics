import h5py

def inspect_hdf5_file(filename='Features/No_ICA_subject_eeg_features_S9_hierarchical.h5'):
    """
    Print the structure of the HDF5 file, including groups and datasets.
    """
    with h5py.File(filename, 'r') as h5f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"📂 Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"📄 Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

        print(f"\n🔍 Inspecting HDF5 File: {filename}\n")
        h5f.visititems(print_structure)

inspect_hdf5_file()
