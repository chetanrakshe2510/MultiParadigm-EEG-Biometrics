# In manual_segmentation_gui/segmentation_gui.py

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import mne

class EEGSegmenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Manual Segmenter")
        self.root.geometry("600x300")

        self.file_path_var = tk.StringVar(value="No file loaded.")
        self.raw = None

        # --- Widgets ---
        tk.Label(root, text="Manual EEG Segmenter", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # File Selection Frame
        file_frame = tk.Frame(root)
        file_frame.pack(pady=5, fill=tk.X, padx=20)
        tk.Button(file_frame, text="Load VHDR File", command=self.load_file).pack(side=tk.LEFT)
        tk.Label(file_frame, textvariable=self.file_path_var, fg="blue", wraplength=400).pack(side=tk.LEFT, padx=10)

        # Main Action Buttons
        tk.Button(root, text="Visualize & Annotate Data", command=self.visualize_data, font=("Helvetica", 12)).pack(pady=10, fill=tk.X, padx=20)
        tk.Button(root, text="Save Annotated Segments", command=self.process_segments, font=("Helvetica", 12)).pack(pady=5, fill=tk.X, padx=20)
        tk.Button(root, text="Exit", command=self.exit_app, fg="red").pack(pady=20)

        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select a BrainVision Header File",
            filetypes=[("BrainVision Files", "*.vhdr")]
        )
        if not file_path:
            return
        
        try:
            self.raw = mne.io.read_raw_brainvision(file_path, preload=True)
            self.file_path_var.set(Path(file_path).name)
            messagebox.showinfo("Success", "EEG data loaded successfully!\nYou can now visualize and annotate.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load EEG file:\n{e}")
            self.raw = None
            self.file_path_var.set("Failed to load file.")

    def visualize_data(self):
        if self.raw is None:
            messagebox.showwarning("No Data", "Please load an EEG file first.")
            return
        
        messagebox.showinfo(
            "Instructions", 
            "The MNE plot window will now open.\n\n"
            "1. Drag with the mouse to select a region.\n"
            "2. Press 'a' to open the annotation dialog.\n"
            "3. Enter a descriptive name (e.g., 'Motor_Task' or 'Eyes_Open') and click OK.\n\n"
            "Close the plot window when you are finished."
        )
        self.raw.plot(block=True) # block=True makes the app wait until the plot is closed

    def process_segments(self):
        if self.raw is None or not self.raw.annotations:
            messagebox.showwarning("No Annotations", "Please load a file and add annotations first.")
            return

        save_dir = filedialog.askdirectory(title="Select a Folder to Save Your Segments")
        if not save_dir:
            return

        count = 0
        for annot in self.raw.annotations:
            description = annot['description']
            onset = annot['onset']
            duration = annot['duration']
            
            # Clean the description to make a valid filename
            clean_desc = "".join(c for c in description if c.isalnum() or c in ('_', '-')).strip()
            if not clean_desc:
                clean_desc = f"segment_{count+1}"

            segment = self.raw.copy().crop(tmin=onset, tmax=onset + duration)
            
            # Suggest a filename based on the original file and the annotation
            original_stem = Path(self.raw.filenames[0]).stem
            output_filename_base = f"{original_stem}_{clean_desc}"
            output_path = Path(save_dir) / f"{output_filename_base}.vhdr"

            segment.export(output_path, fmt="brainvision", overwrite=True)
            count += 1
        
        messagebox.showinfo("Success", f"Successfully saved {count} annotated segment(s) to:\n{save_dir}")

    def exit_app(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to exit?"):
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGSegmenterApp(root)
    root.mainloop()