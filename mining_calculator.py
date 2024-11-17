import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io

# Cache the model loading so it only loads once
@st.cache_resource
def load_models():
    loaded_model = joblib.load('xgboost_mining_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    loaded_le = joblib.load('label_encoder.joblib')
    return loaded_model, loaded_scaler, loaded_le

def clean_numeric(x):
    return float(str(x).strip().replace(',', '.'))

def predict_mining_class(features_batch, model, scaler, le):
    # Batch processing instead of row by row
    features_scaled = scaler.transform(features_batch)
    predictions_encoded = model.predict(features_scaled)
    predictions = le.inverse_transform(predictions_encoded)
    return predictions

@st.cache_data
def process_file(input_file, _model, _scaler, _le):  # Added underscores to model arguments
    # Read the uploaded file based on file extension
    file_extension = input_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        input_data = pd.read_csv(input_file)
    elif file_extension == 'xlsx':
        input_data = pd.read_excel(input_file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")
    
    # Rest of the function remains the same
    input_data.iloc[:, 0] = pd.to_numeric(input_data.iloc[:, 0].str.strip().str.replace(',', '.'), errors='coerce')
    input_data.iloc[:, 1] = pd.to_numeric(input_data.iloc[:, 1].str.strip().str.replace(',', '.'), errors='coerce')
    
    features = input_data.iloc[:, :2].values
    predictions = predict_mining_class(features, _model, _scaler, _le)  # Updated argument names here too
    input_data['Predicted_Class'] = predictions
    
    return input_data

@st.cache_data
def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.read()

def main():
    st.title('Mining Class Predictor')    
    # Load the models
    try:
        model, scaler, le = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # File upload
    st.write("Please upload your CSV or XLSX file with only two columns with no headers(PLI, SPACE)")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Add a spinner during processing
            with st.spinner('Processing your file...'):
                results = process_file(uploaded_file, model, scaler, le)
            
            # Show preview of results
            st.write("Preview of processed data:")
            st.dataframe(results.head())
            
            # Create download button
            excel_data = to_excel(results)
            st.download_button(
                label="Download Excel file",
                data=excel_data,
                file_name="predictions_output.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your CSV file has the correct format with two columns.")
    
    # Add some vertical space before the credit line
    st.markdown("<br>" * 5, unsafe_allow_html=True)
    
    # Add credit line at the bottom
    st.markdown("<h4 style='text-align: right; color: grey;'>Made by Rizky Azmi Swandy (rizkyswandy@gmail.com)</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()