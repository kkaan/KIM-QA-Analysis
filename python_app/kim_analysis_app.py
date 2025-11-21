import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
import os
import sys
import pandas as pd
import numpy as np

# Ensure we can import the logic module
sys.path.append(os.path.join(os.path.dirname(__file__)))
from kim_analysis_logic import (
    parse_centroid_file, parse_trajectory_file, calculate_metrics,
    parse_kim_data, parse_couch_shifts, parse_robot_file, process_interrupt_data
)

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class KimAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("KIM QA Analysis")
        self.geometry("1200x800")

        # --- Data Storage ---
        # Static Analysis
        self.centroid_data = None
        self.trajectory_data = None
        self.trajectory_filename = ""
        self.centroid_filename = ""
        self.span_selector = None
        
        # Interrupt Analysis
        self.int_centroid_data = None # Re-use or separate? Let's keep separate for clarity
        self.int_traj_folder = ""
        self.int_robot_file = ""
        self.int_couch_file = ""
        self.int_kim_data = None
        self.int_robot_data = None
        self.int_shifts = None

        # --- Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Tab View
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.tab_static = self.tabview.add("Static Analysis")
        self.tab_interrupt = self.tabview.add("Interrupt Analysis")
        
        # Initialize Tabs
        self.setup_static_tab()
        self.setup_interrupt_tab()

    def setup_static_tab(self):
        tab = self.tab_static
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Sidebar (Static)
        sidebar = ctk.CTkFrame(tab, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(sidebar, text="Static Analysis", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        ctk.CTkButton(sidebar, text="Load Centroid File", command=self.load_centroid_file).grid(row=1, column=0, padx=20, pady=10)
        self.centroid_label = ctk.CTkLabel(sidebar, text="No Centroid File Loaded", text_color="gray", wraplength=200)
        self.centroid_label.grid(row=2, column=0, padx=20, pady=(0, 10))

        ctk.CTkButton(sidebar, text="Load Trajectory File", command=self.load_trajectory_file).grid(row=3, column=0, padx=20, pady=10)
        self.traj_label = ctk.CTkLabel(sidebar, text="No Trajectory File Loaded", text_color="gray", wraplength=200)
        self.traj_label.grid(row=4, column=0, padx=20, pady=(0, 10))

        ctk.CTkLabel(sidebar, text="-----------------").grid(row=5, column=0, padx=20, pady=5)

        ctk.CTkLabel(sidebar, text="Start Time (s):").grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        self.time_start_entry = ctk.CTkEntry(sidebar)
        self.time_start_entry.grid(row=7, column=0, padx=20, pady=(0, 10))
        self.time_start_entry.insert(0, "10")

        ctk.CTkLabel(sidebar, text="End Time (s):").grid(row=8, column=0, padx=20, pady=(0, 0), sticky="w")
        self.time_end_entry = ctk.CTkEntry(sidebar)
        self.time_end_entry.grid(row=9, column=0, padx=20, pady=(0, 10))

        ctk.CTkButton(sidebar, text="Calculate Metrics", command=self.calculate_metrics_action, fg_color="green").grid(row=10, column=0, padx=20, pady=20, sticky="s")
        ctk.CTkButton(sidebar, text="Save Results", command=self.save_results_action).grid(row=11, column=0, padx=20, pady=20, sticky="s")

        # Main Content (Static)
        main_frame = ctk.CTkFrame(tab, corner_radius=0)
        main_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Plot
        self.plot_frame_static = ctk.CTkFrame(main_frame)
        self.plot_frame_static.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.fig_static, self.ax_static = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas_static = FigureCanvasTkAgg(self.fig_static, master=self.plot_frame_static)
        self.canvas_static.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_static, self.plot_frame_static).update()

        # Metrics
        self.metrics_frame_static = ctk.CTkTextbox(main_frame, height=150)
        self.metrics_frame_static.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        self.metrics_frame_static.insert("0.0", "Metrics will appear here...")
        self.metrics_frame_static.configure(state="disabled")

    def setup_interrupt_tab(self):
        tab = self.tab_interrupt
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_rowconfigure(0, weight=1)

        # Sidebar (Interrupt)
        sidebar = ctk.CTkFrame(tab, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(sidebar, text="Interrupt Analysis", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        # Inputs
        ctk.CTkButton(sidebar, text="Load Trajectory Folder", command=self.load_int_folder).grid(row=1, column=0, padx=20, pady=10)
        self.int_folder_label = ctk.CTkLabel(sidebar, text="No Folder Selected", text_color="gray", wraplength=200)
        self.int_folder_label.grid(row=2, column=0, padx=20, pady=(0, 10))

        ctk.CTkButton(sidebar, text="Load Robot File", command=self.load_int_robot).grid(row=3, column=0, padx=20, pady=10)
        self.int_robot_label = ctk.CTkLabel(sidebar, text="No Robot File Loaded", text_color="gray", wraplength=200)
        self.int_robot_label.grid(row=4, column=0, padx=20, pady=(0, 10))
        
        # Couch Shift file is usually inside the folder, but let's allow manual override or auto-detect
        self.int_couch_label = ctk.CTkLabel(sidebar, text="Couch Shifts: Auto-detect", text_color="gray", wraplength=200)
        self.int_couch_label.grid(row=5, column=0, padx=20, pady=(0, 10))

        ctk.CTkButton(sidebar, text="Analyze Interrupt", command=self.analyze_interrupt_action, fg_color="green").grid(row=10, column=0, padx=20, pady=20, sticky="s")
        ctk.CTkButton(sidebar, text="Save Results", command=self.save_int_results_action).grid(row=11, column=0, padx=20, pady=20, sticky="s")

        # Main Content (Interrupt)
        main_frame = ctk.CTkFrame(tab, corner_radius=0)
        main_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        # Plot
        self.plot_frame_int = ctk.CTkFrame(main_frame)
        self.plot_frame_int.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.fig_int, self.ax_int = plt.subplots(figsize=(6, 4), dpi=100)
        self.canvas_int = FigureCanvasTkAgg(self.fig_int, master=self.plot_frame_int)
        self.canvas_int.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas_int, self.plot_frame_int).update()

        # Metrics
        self.metrics_frame_int = ctk.CTkTextbox(main_frame, height=150)
        self.metrics_frame_int.grid(row=1, column=0, padx=20, pady=20, sticky="ew")
        self.metrics_frame_int.insert("0.0", "Metrics will appear here...")
        self.metrics_frame_int.configure(state="disabled")

    # --- Static Analysis Methods ---
    def load_centroid_file(self):
        filename = filedialog.askopenfilename(title="Select Centroid File", filetypes=[("Text Files", "*.txt")])
        if filename:
            try:
                self.centroid_data = parse_centroid_file(filename)
                self.centroid_filename = filename
                self.centroid_label.configure(text=os.path.basename(filename), text_color="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse centroid file: {e}")

    def load_trajectory_file(self):
        filename = filedialog.askopenfilename(title="Select Trajectory File", filetypes=[("Text/CSV Files", "*.txt *.csv *.his")])
        if filename:
            try:
                self.trajectory_data = parse_trajectory_file(filename)
                self.trajectory_filename = filename
                self.traj_label.configure(text=os.path.basename(filename), text_color="green")
                
                max_time = self.trajectory_data['time'].max()
                self.time_end_entry.delete(0, tk.END)
                self.time_end_entry.insert(0, str(round(max_time, 2)))
                
                self.plot_static_data()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to parse trajectory file: {e}")

    def plot_static_data(self):
        if self.trajectory_data is None: return
        self.ax_static.clear()
        
        if self.centroid_data:
            df_dev, _ = calculate_metrics(self.trajectory_data, self.centroid_data['expected_centroid'])
            self.ax_static.plot(df_dev['time'], df_dev['dev_x'], label='LR (X) Dev')
            self.ax_static.plot(df_dev['time'], df_dev['dev_y'], label='SI (Y) Dev')
            self.ax_static.plot(df_dev['time'], df_dev['dev_z'], label='AP (Z) Dev')
            self.ax_static.set_ylabel("Deviation (mm)")
        else:
            self.ax_static.plot(self.trajectory_data['time'], self.trajectory_data['meas_x'], label='LR (X) Raw')
            self.ax_static.plot(self.trajectory_data['time'], self.trajectory_data['meas_y'], label='SI (Y) Raw')
            self.ax_static.plot(self.trajectory_data['time'], self.trajectory_data['meas_z'], label='AP (Z) Raw')
            self.ax_static.set_ylabel("Position (mm)")

        self.ax_static.set_xlabel("Time (s)")
        self.ax_static.legend()
        self.ax_static.grid(True)
        
        if 'gantry' in self.trajectory_data.columns:
            ax2 = self.ax_static.twiny()
            gantry_min = self.trajectory_data['gantry'].min()
            gantry_max = self.trajectory_data['gantry'].max()
            start_angle = np.ceil(gantry_min / 10) * 10
            end_angle = np.floor(gantry_max / 10) * 10
            target_angles = np.arange(start_angle, end_angle + 1, 10)
            
            tick_locs = []
            tick_labels = []
            for angle in target_angles:
                idx = (self.trajectory_data['gantry'] - angle).abs().idxmin()
                row = self.trajectory_data.loc[idx]
                if abs(row['gantry'] - angle) < 1.0:
                    tick_locs.append(row['time'])
                    tick_labels.append(str(int(angle)))
            
            ax2.set_xlim(self.ax_static.get_xlim())
            ax2.set_xticks(tick_locs)
            ax2.set_xticklabels(tick_labels)
            ax2.set_xlabel("Gantry Angle (°)")
            
            def update_ax2(event):
                ax2.set_xlim(self.ax_static.get_xlim())
                self.canvas_static.draw_idle()
            self.ax_static.callbacks.connect("xlim_changed", update_ax2)

        self.span_selector = SpanSelector(
            self.ax_static, self.on_select_static, 'horizontal', useblit=True,
            props=dict(alpha=0.5, facecolor='red'), interactive=True, drag_from_anywhere=True
        )
        self.canvas_static.draw()

    def on_select_static(self, xmin, xmax):
        self.time_start_entry.delete(0, tk.END)
        self.time_start_entry.insert(0, f"{xmin:.2f}")
        self.time_end_entry.delete(0, tk.END)
        self.time_end_entry.insert(0, f"{xmax:.2f}")

    def calculate_metrics_action(self):
        if self.trajectory_data is None or self.centroid_data is None:
            messagebox.showwarning("Warning", "Please load both Centroid and Trajectory files.")
            return
        try:
            t_start = float(self.time_start_entry.get())
            t_end = float(self.time_end_entry.get())
            df_res, metrics = calculate_metrics(self.trajectory_data, self.centroid_data['expected_centroid'], (t_start, t_end))
            
            result_text = f"Analysis Interval: {t_start:.2f}s - {t_end:.2f}s\n\n"
            result_text += f"{'Axis':<10} {'Mean (mm)':<15} {'Std (mm)':<15} {'5% (mm)':<15} {'95% (mm)':<15}\n"
            result_text += "-" * 70 + "\n"
            for axis in ['x', 'y', 'z']:
                axis_label = "LR" if axis == 'x' else "SI" if axis == 'y' else "AP"
                result_text += f"{axis_label:<10} {metrics[f'mean_{axis}']:<15.2f} {metrics[f'std_{axis}']:<15.2f} {metrics[f'p5_{axis}']:<15.2f} {metrics[f'p95_{axis}']:<15.2f}\n"
            
            self.metrics_frame_static.configure(state="normal")
            self.metrics_frame_static.delete("0.0", tk.END)
            self.metrics_frame_static.insert("0.0", result_text)
            self.metrics_frame_static.configure(state="disabled")
            self.last_metrics_static = metrics
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {e}")

    def save_results_action(self):
        if not hasattr(self, 'last_metrics_static'):
            messagebox.showwarning("Warning", "Please calculate metrics first.")
            return
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if save_dir:
            try:
                with open(os.path.join(save_dir, "Metrics.txt"), 'w') as f:
                    f.write(self.metrics_frame_static.get("0.0", tk.END))
                self.fig_static.savefig(os.path.join(save_dir, "Trace_Plot.png"))
                messagebox.showinfo("Success", f"Results saved to {save_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")

    # --- Interrupt Analysis Methods ---
    def load_int_folder(self):
        folder = filedialog.askdirectory(title="Select Trajectory Folder")
        if folder:
            self.int_traj_folder = folder
            self.int_folder_label.configure(text=os.path.basename(folder), text_color="green")
            
            # Auto-detect couch shifts
            couch_file = os.path.join(folder, "couchShifts.txt")
            if os.path.exists(couch_file):
                self.int_couch_file = couch_file
                self.int_couch_label.configure(text="Couch Shifts: Found", text_color="green")
            else:
                self.int_couch_label.configure(text="Couch Shifts: Not Found", text_color="red")

    def load_int_robot(self):
        filename = filedialog.askopenfilename(title="Select Robot File", filetypes=[("Text Files", "*.txt")])
        if filename:
            self.int_robot_file = filename
            self.int_robot_label.configure(text=os.path.basename(filename), text_color="green")

    def analyze_interrupt_action(self):
        if not self.int_traj_folder or not self.int_robot_file:
            messagebox.showwarning("Warning", "Please load Trajectory Folder and Robot File.")
            return
        
        if not self.int_couch_file:
             messagebox.showwarning("Warning", "Couch Shifts file not found in folder.")
             return

        try:
            # Parse Data
            kim_df = parse_kim_data(self.int_traj_folder)
            shifts = parse_couch_shifts(self.int_couch_file)
            robot_df = parse_robot_file(self.int_robot_file)
            
            if kim_df.empty:
                raise ValueError("No KIM data found in folder.")
            if robot_df.empty:
                raise ValueError("Robot file empty or invalid.")
            
            # Process
            processed_df, metrics = process_interrupt_data(kim_df, robot_df, shifts)
            
            if processed_df is None:
                raise ValueError("Processing failed.")
                
            self.int_kim_data = processed_df
            self.last_metrics_int = metrics
            
            # Plot
            self.ax_int.clear()
            
            # Plot KIM vs Robot (Aligned)
            # KIM
            self.ax_int.plot(processed_df['time'], processed_df['meas_x'], 'b.', label='LR (KIM)', markersize=2)
            self.ax_int.plot(processed_df['time'], processed_df['meas_y'], 'g.', label='SI (KIM)', markersize=2)
            self.ax_int.plot(processed_df['time'], processed_df['meas_z'], 'r.', label='AP (KIM)', markersize=2)
            
            # Robot (Interpolated/Aligned)
            self.ax_int.plot(processed_df['time'], processed_df['robot_x'], 'b-', label='LR (Robot)', alpha=0.5)
            self.ax_int.plot(processed_df['time'], processed_df['robot_y'], 'g-', label='SI (Robot)', alpha=0.5)
            self.ax_int.plot(processed_df['time'], processed_df['robot_z'], 'r-', label='AP (Robot)', alpha=0.5)
            
            self.ax_int.set_xlabel("Time (s)")
            self.ax_int.set_ylabel("Position (mm)")
            self.ax_int.legend(loc='upper right', fontsize='small', ncol=2)
            self.ax_int.grid(True)
            self.canvas_int.draw()
            
            # Display Metrics
            result_text = "Interrupt Analysis Results\n"
            result_text += f"{'Axis':<10} {'Mean (mm)':<15} {'Std (mm)':<15} {'5% (mm)':<15} {'95% (mm)':<15}\n"
            result_text += "-" * 70 + "\n"
            
            # Check Pass/Fail
            pass_fail = []
            
            for axis in ['lr', 'si', 'ap']:
                mean_val = metrics[f'mean_{axis}']
                std_val = metrics[f'std_{axis}']
                p5 = metrics[f'p5_{axis}']
                p95 = metrics[f'p95_{axis}']
                
                result_text += f"{axis.upper():<10} {mean_val:<15.2f} {std_val:<15.2f} {p5:<15.2f} {p95:<15.2f}\n"
                
                if abs(mean_val) > 1.0 or std_val > 2.0:
                    pass_fail.append(f"{axis.upper()} FAIL")
            
            result_text += "\n" + ("PASS" if not pass_fail else f"FAIL: {', '.join(pass_fail)}")
            
            self.metrics_frame_int.configure(state="normal")
            self.metrics_frame_int.delete("0.0", tk.END)
            self.metrics_frame_int.insert("0.0", result_text)
            self.metrics_frame_int.configure(state="disabled")

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

    def save_int_results_action(self):
        if not hasattr(self, 'last_metrics_int'):
            messagebox.showwarning("Warning", "Please analyze first.")
            return
        save_dir = filedialog.askdirectory(title="Select Directory to Save Results")
        if save_dir:
            try:
                with open(os.path.join(save_dir, "Interrupt_Metrics.txt"), 'w') as f:
                    f.write(self.metrics_frame_int.get("0.0", tk.END))
                self.fig_int.savefig(os.path.join(save_dir, "Interrupt_Plot.png"))
                messagebox.showinfo("Success", f"Results saved to {save_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {e}")

if __name__ == "__main__":
    app = KimAnalysisApp()
    app.mainloop()
