import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for plots
plt.style.use('seaborn')

# Cache the model loading so it only loads once
@st.cache_resource
def load_models():
    loaded_model = joblib.load('xgboost_mining_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    loaded_le = joblib.load('label_encoder.joblib')
    return loaded_model, loaded_scaler, loaded_le

def create_scatter_plot(df, sample_size=10):
    """Create a scatter plot of PLI vs SPACE with classifications"""
    # Sample random rows if dataset is larger than sample_size
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42)
    else:
        plot_df = df
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot with different colors for each classification
    for classification in plot_df['CLASSIFICATION'].unique():
        mask = plot_df['CLASSIFICATION'] == classification
        ax.scatter(
            plot_df[mask]['PLI'], 
            plot_df[mask]['SPACE'],
            label=classification,
            alpha=0.6
        )
    
    plt.xlabel('PLI')
    plt.ylabel('SPACE')
    plt.title('PLI vs SPACE Classification (Random 10 Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return fig

def detect_headers(df):
    """Check if the DataFrame has the expected headers (PLI, SPACE)"""
    expected_headers = {'PLI', 'SPACE'}
    if len(df.columns) >= 2:
        first_row_headers = {str(col).strip().upper() for col in df.columns[:2]}
        return bool(expected_headers & first_row_headers)
    return False

def clean_and_prepare_data(df):
    """Clean and prepare the data, ensuring correct column names and types"""
    # If df has more than 2 columns, keep only the first two
    df = df.iloc[:, :2]
    
    # Convert columns to numeric, replacing commas with periods
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(
            df[col].str.replace(',', '.'), 
            errors='coerce'
        )
    
    # Rename columns to standard names
    df.columns = ['PLI', 'SPACE']
    return df

def predict_mining_class(features_df, model, scaler, le):
    """Make predictions using the loaded models"""
    features_scaled = scaler.transform(features_df)
    predictions_encoded = model.predict(features_scaled)
    predictions = le.inverse_transform(predictions_encoded)
    return predictions

@st.cache_data
def process_file(input_file, _model, _scaler, _le):
    # Read the uploaded file based on file extension
    file_extension = input_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'csv':
            input_data = pd.read_csv(input_file)
            if not detect_headers(input_data):
                input_data = pd.read_csv(input_file, header=None)
        elif file_extension == 'xlsx':
            input_data = pd.read_excel(input_file, engine='openpyxl')
            if not detect_headers(input_data):
                input_data = pd.read_excel(input_file, header=None, engine='openpyxl')
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or XLSX file.")
        
        # Clean and prepare the data
        input_data = clean_and_prepare_data(input_data)
        
        # Make predictions
        predictions = predict_mining_class(input_data, _model, _scaler, _le)
        
        # Create output DataFrame with standardized column names
        output_data = pd.DataFrame({
            'PLI': input_data['PLI'],
            'SPACE': input_data['SPACE'],
            'CLASSIFICATION': predictions
        })
        
        return output_data
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        raise e

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
    st.write("Please upload your CSV or XLSX file. The file should contain two columns: PLI and SPACE")
    st.write("If your file doesn't have headers, the first two columns will be used as PLI and SPACE respectively")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Add a spinner during processing
            with st.spinner('Processing your file...'):
                results = process_file(uploaded_file, model, scaler, le)
            
            # Show preview of results
            st.write("Preview of first 5 rows:")
            st.dataframe(results.head())
            
            # Show random 10 samples
            st.write("Random 10 samples:")
            random_samples = results.sample(n=min(10, len(results)), random_state=42)
            st.dataframe(random_samples)
            
            # Create and show visualization
            st.write("Visualization of random samples:")
            fig = create_scatter_plot(results)
            st.pyplot(fig)
            
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
            st.write("Please make sure your file has the correct format with two columns.")
    
    # Add some vertical space before the credit line
    st.markdown("<br>" * 5, unsafe_allow_html=True)
    
    # Add credit line at the bottom
    st.markdown("<h4 style='text-align: right; color: grey;'>Made by Rizky Azmi Swandy (rizkyswandy@gmail.com)</h4>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()