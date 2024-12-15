import onnxruntime as ort
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import time  # Import time for timing predictions

# ASCII Art (ในกรณีนี้จะใช้ฟอนต์ที่พิมพ์ใน console)
ascii_art = r"""
 .----------------.  .----------------.  .-----------------. .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| | ____    ____ | || |     _____    | || | ____  _____  | || |  _________   | |
| ||_   \  /   _|| || |    |_   _|   | || ||_   \|_   _| | || | |  _   _  |  | | 
| |  |   \/   |  | || |      | |     | || |  |   \ | |   | || | |_/ | | \_|  | |
| |  | |\  /| |  | || |      | |     | || |  | |\ \| |   | || |     | |      | |
| | _| |_\/_| |_ | || |     _| |_    | || | _| |_\   |_  | || |    _| |_     | |
| ||_____||_____|| || |    |_____|   | || ||_____|\____| | || |   |_____|    | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
"""

def check_model_file():
    current_directory = os.getcwd()
    model_path = os.path.join(current_directory, "large_model.onnx")
    if not os.path.exists(model_path):        
        print("Please ensure the model file is in the same directory as this script.")
        sys.exit(1)
    return model_path

def load_model(model_path):
    return ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

def predict_magnetization(session, thickness, diameter, temperature):
    start_time = time.time()  # Start timing
    input_data = np.array([[thickness, diameter, temperature]], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_data})
    end_time = time.time()  # End timing
    prediction_time = end_time - start_time  # Calculate prediction time
    return result[0][0][0], prediction_time  # Return both result and prediction time

# ฟังก์ชันสำหรับ M(T) = (1 - T/Tc) ^ beta
def magnetization_model(T, Tc, beta):
    return np.where(T < Tc, np.nan_to_num((1 - T/Tc) ** beta), 0)

def auto_mode(session, output_dir):
    thicknesses = list(map(float, input("Enter thicknesses separated by commas: ").split(',')))
    
    while True:
        try:
            diameters = list(map(float, input("Enter diameters separated by commas: ").split(',')))
            break
        except ValueError:
            print("Invalid input. Please enter diameters as a comma-separated list of numbers (e.g., 1,2,3).")

    # Create a figure for all plots
    plt.figure(figsize=(12, 8))

    for thickness in thicknesses:
        for diameter in diameters:
            temperatures = np.arange(0, 901, 2)
            magnetizations = []
            prediction_times = []  # List to store prediction times

            for temp in temperatures:
                mag, pred_time = predict_magnetization(session, thickness, diameter, temp)
                magnetizations.append(mag)
                prediction_times.append(pred_time)  # Store the prediction time

            magnetization_smooth = savgol_filter(magnetizations, window_length=11, polyorder=2)
            
            # Save smoothed data to CSV
            filename = os.path.join(output_dir, f"small_rf_model-data_t{thickness}d{diameter}_small-datasets.csv")
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Temperature", "Magnetization_smooth"])
                writer.writerows(zip(temperatures, magnetization_smooth))
            print(f"Smoothed data saved to {filename}")

            label = f'thickness={thickness}, diameter={diameter}'
            plt.plot(temperatures, magnetization_smooth, '-', label=f'AI Predicted {label}', linewidth=7, alpha=0.5)

            # ฟิตข้อมูลเพื่อหาค่า Tc และ beta
            try:
                initial_guess = [max(temperatures) * 0.8, 1]
                bounds = ([1, 0], [np.inf, np.inf])
                params, _ = curve_fit(magnetization_model, temperatures, magnetization_smooth, p0=initial_guess, bounds=bounds)
                Tc, beta = params
                print(f"Curie Temperature (Tc): {Tc:.2f} K for {label}")
                print(f"Beta: {beta:.2f}")
                avg_prediction_time = np.mean(prediction_times)
                print(f"Average Prediction Time: {avg_prediction_time:.4f} seconds")

                temperature_fit = np.linspace(min(temperatures), max(temperatures), 100)
                magnetization_fit = magnetization_model(temperature_fit, Tc, beta)

                # Plot the fitted curve
                plt.plot(temperature_fit, magnetization_fit, '--', color='blue', label=f'Tc={Tc:.2f}, beta={beta:.2f}', linewidth=3)

                # Add a red dashed line for Tc
                plt.axvline(x=Tc, color='red', linestyle='--', linewidth=2)
             
            except Exception as e:
                print(f"Error in fitting data: {e}")

    # ปรับแต่งกราฟแยกสำหรับแต่ละคู่
    plt.xlabel('Temperature (K)', fontsize=20)
    plt.ylabel(r'Magnetization (M/M$\mathregular{_s}$)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # Add average prediction time to the title
    avg_prediction_time = np.mean(prediction_times)
    plt.title(f'Predicted Curie Temperature\nPrediction Time: {avg_prediction_time:.4f} seconds', fontsize=20)
    
    plt.legend(fontsize=17)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(output_dir, "fitted_rf_model_data_combined.png"))
    plt.show()

def main_menu():
    print("Magnetic Information Storage Technology, MSU & UoY")
    print(ascii_art)
    print("Checking for ONNX model file...")
    model_path = check_model_file()
    print("Loading ONNX model. Please wait...")
    session = load_model(model_path)
    
    root = tk.Tk()
    root.withdraw()
    output_dir = filedialog.askdirectory(title="Select Directory to Save Processed Files")
    
    while True:
        print("\nMenu:")
        print("(1) Run auto mode, Temperature 0 to 900 with smoothed data and curve fitting")
        print("(2) Exit")
        
        choice = input("Enter your choice (1-2): ")
        
        if choice == '1':
            auto_mode(session, output_dir)
        elif choice == '2':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
