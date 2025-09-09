import streamlit as st
import pandas as pd
from io import BytesIO
import os
import google.generativeai as genai
import tempfile
import importlib
import subprocess
import sys
import matplotlib.pyplot as plt

try:
    import xlsxwriter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
    import xlsxwriter

# Configure API key
GOOGLE_API_KEY = "AIzaSyCDwcPcdKq8J5KUfJ6RjdaRQn5ZuLkuuco"
try:
    api_key = os.getenv("GOOGLE_API_KEY") or GOOGLE_API_KEY
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"API configuration failed: {e}")
    st.info("Please get a valid API key from: https://makersuite.google.com/app/apikey")
    st.stop()

def load_excel(file):
    df = pd.read_excel(file)
    return df

def normalize_percentages(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name] * 100
    return df



def extract_python_code(text):
    """Extract Python code from markdown-formatted text."""
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        return text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        return text[start:end].strip()
    return text.strip()

def generate_python_code(user_query, df_columns):
    try:
        prompt = f"""
Generate only Python code (no explanations or markdown) for a function named `process_dataframe_query` that:
- Takes parameters: df (DataFrame), query (string)
- Processes the DataFrame based on: {user_query}
- Available columns: {', '.join(df_columns)}
- Returns: DataFrame, number, or matplotlib figure
- Import required modules (pandas as pd, matplotlib.pyplot as plt)
        """
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        code = extract_python_code(response.text)
        return code
    except Exception as e:
        return f"Error: {str(e)}"

def execute_code_query(df, user_query):
    code = generate_python_code(user_query, df.columns)
    
    if code.startswith("Error:"):
        return code
    
    file_path = "generated_code.py"
    try:
        with open(file_path, "w") as f:
            f.write(code)

        spec = importlib.util.spec_from_file_location("generated_module", file_path)
        generated_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(generated_module)
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[1]
            subprocess.check_call([sys.executable, "-m", "pip", "install", missing_module])
            spec.loader.exec_module(generated_module)

        result = generated_module.process_dataframe_query(df, user_query)

        os.remove(file_path)  # Clean up the generated code file
        return result

    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except AttributeError:
        return "Error: The generated code must define a 'process_dataframe_query' function."
    except SyntaxError as e:
        return f"Syntax error in generated code: {e}"
    except Exception as e:
        return f"Error executing generated code: {e}"

def save_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

# Main Streamlit application
def main():
    st.set_page_config(page_title="Excel Query Chatbot with AI", layout="wide")
    st.title("Excel Query Chatbot with AI")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file:
        df = load_excel(uploaded_file)
        df = normalize_percentages(df, "Progress")
        st.write("Data Preview:", df.head())

        user_query = st.text_input("Ask a question about the data")
        if user_query:
            with st.spinner("Processing your query..."):
                response2 = execute_code_query(df, user_query)
                st.write("Response:", response2)

                if isinstance(response2, pd.DataFrame):
                    st.download_button(
                        label="Download Filtered Data",
                        data=save_to_excel(response2),
                        file_name="filtered_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

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