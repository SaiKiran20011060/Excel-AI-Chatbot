import streamlit as st
import pandas as pd
from io import BytesIO
import os
import importlib.util
import subprocess
import sys
import re
import requests
import matplotlib.pyplot as plt

# Ensure xlsxwriter is installed
try:
    import xlsxwriter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
    import xlsxwriter

def load_excel(file):
    return pd.read_excel(file)

def normalize_percentages(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name] * 100
    return df

def extract_python_code(raw_text):
    """Extracts raw python code from markdown code blocks or raw text."""
    pattern = r"```(?:python)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def generate_python_code(user_query, df_columns, api_key):
    prompt = f"""
You are a Python data analysis and visualization expert. Generate a Python script that processes a Pandas DataFrame.

Requirements:
1. Define a function named `process_dataframe_query(df, query)`.
2. `df` is a Pandas DataFrame, `query` is a string.
3. Return ONE of the following:
   - A filtered/modified Pandas DataFrame.
   - A summary scalar statistic (int, float, str).
   - A Matplotlib figure object (`fig = plt.gcf()`).

DataFrame Columns available: {', '.join(df_columns)}

User Query: {user_query}

Return ONLY executable Python code inside a markdown code block (```python ... ```). Do not add conversational text outside the code block.
"""

    models_to_try = ["gemini-3.5-flash", "gemini-2.5-flash"]
    errors = []

    for model_name in models_to_try:
        try:
            # FIX: Appending the API key directly to the URL as a query parameter
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key.strip()}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                try:
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return extract_python_code(response_text)
                except (KeyError, IndexError) as e:
                    errors.append(f"{model_name}: Failed to parse JSON - {e}")
            else:
                errors.append(f"{model_name}: HTTP {response.status_code} - {response.text}")
                
        except Exception as e:
            errors.append(f"{model_name}: {str(e)}")

    return f"Error generating code across models:\n" + "\n".join(errors)

def execute_code_query(df, user_query, api_key):
    code = generate_python_code(user_query, df.columns, api_key)

    if code.startswith("Error"):
        return code

    file_path = "generated_code.py"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location("generated_module", file_path)
        generated_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_module)

        result = generated_module.process_dataframe_query(df, user_query)

        if os.path.exists(file_path):
            os.remove(file_path)
        return result

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return f"Error executing generated code: {e}\n\nGenerated Code:\n```python\n{code}\n```"

def save_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

def main():
    st.set_page_config(page_title="Excel Query Chatbot with AI", layout="wide")
    st.title("Excel Query Chatbot with AI")

    # ==========================================
    # PASTE YOUR NEW AQ. API KEY HERE
    # ==========================================
    api_key = "AIzaSyDh3ITY742S6s9mQHPP0YNKbJbiboVZmWo"  
    
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = load_excel(uploaded_file)
        df = normalize_percentages(df, "Progress")
        st.write("Data Preview:", df.head())

        user_query = st.text_input("Ask a question about the data")
        if user_query:
            with st.spinner("Processing your query..."):
                response = execute_code_query(df, user_query, api_key)

                if isinstance(response, pd.DataFrame):
                    st.write("Response (DataFrame):", response)
                    st.download_button(
                        label="Download Filtered Data",
                        data=save_to_excel(response),
                        file_name="filtered_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif isinstance(response, plt.Figure):
                    st.pyplot(response)
                else:
                    st.write("Response:", response)

        st.download_button(
            label="Download Updated Excel File",
            data=save_to_excel(df),
            file_name="updated_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Please upload an Excel file to get started.")

if __name__ == "__main__":
    main()
    
