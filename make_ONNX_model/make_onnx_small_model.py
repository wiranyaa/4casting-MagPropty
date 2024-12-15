import re
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import tkinter as tk
from tkinter import filedialog

def process_file(filename, output_dir):
    match = re.match(r't(\d+(\.\d+)?)d(\d+(\.\d+)?)', filename)
    if not match:
        raise ValueError("Invalid filename format")
    
    thickness = float(match.group(1))
    diameter = float(match.group(3))

    df = pd.read_csv(filename, skiprows=6, sep='\t', header=None)
    new_df = pd.DataFrame({
        'thickness': thickness,
        'diameter': diameter,
        'temperature': df.iloc[:, 1],
        'magnetization': df.iloc[:, 6]
    })

    output_filename = os.path.join(output_dir, f"processed_{filename}.csv")
    new_df.to_csv(output_filename, index=False)
    print(f"Processed file saved as {output_filename}")
    return output_filename

def show_working_path():
    current_path = os.getcwd()
    print(f"Current working directory: {current_path}")
    
def combine_files(output_dir):
    files_to_combine = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                        if f.startswith("processed_") and f.endswith(".csv")]
    combined_df = pd.concat([pd.read_csv(f) for f in files_to_combine], ignore_index=True)
    combined_df.to_csv(os.path.join(output_dir, "5-Thickness-data.csv"), index=False)
    print("All processed files combined into 5-Thickness-data.csv")

def delete_processed_files(output_dir):
    for fname in os.listdir(output_dir):
        if fname.startswith('processed_'):
            os.remove(os.path.join(output_dir, fname))
    print("Deleted all files with prefix 'processed_' in the specified directory.")

def train_and_convert_model(X_train, X_test, y_train, y_test, output_dir):
    start_time = time.time()
    
    small_rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    small_rf_model.fit(X_train, y_train)
    
    joblib_path = os.path.join(output_dir, 'small_model.joblib')
    onnx_path = os.path.join(output_dir, 'small_model.onnx')
    
    joblib.dump(small_rf_model, joblib_path)
    
    initial_type = [('float_input', FloatTensorType([None, 3]))]
    # เพิ่มค่า opset ให้ใหม่ขึ้นเพื่อเพิ่มความแม่นยำ
    onx = convert_sklearn(small_rf_model, initial_types=initial_type, target_opset=14)
    
    with open(onnx_path, 'wb') as f:
        f.write(onx.SerializeToString())
    
    compute_time = time.time() - start_time
    
    return small_rf_model, compute_time

def compare_predictions(small_rf_model, X_test, output_dir):
    sklearn_pred = small_rf_model.predict(X_test)
    
    onnx_path = os.path.join(output_dir, "small_model.onnx")
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    X_test_numpy = X_test.to_numpy().astype(np.float32)  # ใช้ float32 เพื่อให้ตรงกับ ONNX
    onnx_pred = sess.run([label_name], {input_name: X_test_numpy})[0]
    
    print("Scikit-learn predictions (first 5):", sklearn_pred[:5])
    print("ONNX predictions (first 5):", onnx_pred[:5])
    
    # คำนวณค่าความต่างระหว่าง Scikit-learn กับ ONNX
    mean_diff = np.mean(np.abs(sklearn_pred - onnx_pred.flatten()))
    print("Mean absolute difference between Scikit-learn and ONNX predictions:", mean_diff)
    
    # เพิ่มการตรวจสอบค่าความแตกต่างสูงสุดและต่ำสุด
    max_diff = np.max(np.abs(sklearn_pred - onnx_pred.flatten()))
    min_diff = np.min(np.abs(sklearn_pred - onnx_pred.flatten()))
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Minimum absolute difference: {min_diff}")

def main():
    show_working_path()
    
    # Open a file dialog to select the output directory
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    output_dir = filedialog.askdirectory(title="Select Directory to Save Processed Files")
    
    if not output_dir:
        print("No directory selected. Exiting...")
        return

    delete_processed_files(output_dir)
    
    for fname in os.listdir(os.getcwd()):
        if fname.startswith("t"):
            process_file(fname, output_dir)
    
    combine_files(output_dir)
    
    print("Loading data...")
    data = pd.read_csv(os.path.join(output_dir, '5-Thickness-data.csv'))
    
    print("Preparing features and target variable...")
    X = data[['thickness', 'diameter', 'temperature']]
    y = data['magnetization']
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the Random Forest Regressor and converting to ONNX...")
    small_rf_model, compute_time = train_and_convert_model(X_train, X_test, y_train, y_test, output_dir)
    print(f"Training and conversion took: {compute_time:.2f} seconds")
    
    print("Comparing predictions...")
    compare_predictions(small_rf_model, X_test, output_dir)

if __name__ == "__main__":
    main()
