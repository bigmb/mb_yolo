import streamlit as st
import yaml
import subprocess
import os

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(file_path, config):
    with open(file_path, 'w') as file:
        yaml.dump(config, file)

def main():
    st.title("YOLO Training Configuration")

    config_path = "config.yaml"
    config = load_config(config_path)

    st.header("Model Configuration")
    config['model'] = st.selectbox("Select YOLO version", ['yolov3', 'yolov5', 'yolov8'], index=['yolov3', 'yolov5', 'yolov8'].index(config['model']))
    config['num_classes'] = st.number_input("Number of classes", min_value=1, value=config['num_classes'])
    config['img_size'] = st.number_input("Image size", min_value=32, value=config['img_size'])

    st.header("Training Configuration")
    config['batch_size'] = st.number_input("Batch size", min_value=1, value=config['batch_size'])
    config['epochs'] = st.number_input("Number of epochs", min_value=1, value=config['epochs'])

    st.header("Data Configuration")
    config['data_yaml'] = st.text_input("Path to data.yaml file", value=config['data_yaml'])

    st.header("Output Configuration")
    config['project'] = st.text_input("Project name", value=config['project'])
    config['name'] = st.text_input("Run name", value=config['name'])

    st.header("Hardware Configuration")
    config['device'] = st.selectbox("Device", ['cuda', 'cpu'], index=['cuda', 'cpu'].index(config['device']))
    config['n_cpu'] = st.number_input("Number of CPU workers", min_value=1, value=config['n_cpu'])

    if st.button("Save Configuration"):
        save_config(config_path, config)
        st.success("Configuration saved successfully!")

    if st.button("Start Training"):
        save_config(config_path, config)
        st.info("Starting training process...")
        try:
            result = subprocess.run(["python", "train.py"], capture_output=True, text=True, check=True)
            st.success("Training completed successfully!")
            st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("An error occurred during training:")
            st.code(e.stderr)

if __name__ == "__main__":
    main()