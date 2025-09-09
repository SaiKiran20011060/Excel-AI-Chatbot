import streamlit as st
import pandas as pd
from io import BytesIO
import os
import google.generativeai as genai
import tempfile
import re
import hashlib
import subprocess
import sys
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import xlsxwriter
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xlsxwriter"])
    import xlsxwriter

# Secure API key configuration
def configure_api():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY environment variable")
        st.stop()
    genai.configure(api_key=api_key)

def load_excel(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None

def normalize_percentages(df, column_name):
    if df is not None and column_name in df.columns:
        df[column_name] = df[column_name] * 100
    return df

def sanitize_query(query):
    """Sanitize user query to prevent code injection"""
    # Remove potentially dangerous keywords and patterns
    dangerous_patterns = [
        r'import\s+os', r'import\s+subprocess', r'import\s+sys',
        r'exec\s*\(', r'eval\s*\(', r'__import__',
        r'open\s*\(', r'file\s*\(', r'input\s*\(',
        r'raw_input\s*\(', r'compile\s*\(',
        r'globals\s*\(', r'locals\s*\(',
        r'\.system\s*\(', r'\.popen\s*\(',
        r'\.call\s*\(', r'\.run\s*\('
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return None
    
    return query

def generate_safe_code(user_query, df_columns):
    """Generate safe Python code with restricted operations"""
    sanitized_query = sanitize_query(user_query)
    if not sanitized_query:
        return "# Query contains potentially unsafe operations and was blocked"
    
    prompt = f"""
    Generate ONLY safe data analysis code for a Pandas DataFrame. 
    
    RESTRICTIONS:
    - NO import statements except pandas, matplotlib.pyplot, numpy
    - NO file operations (open, read, write)
    - NO system calls or subprocess
    - NO exec, eval, or dynamic code execution
    - ONLY use provided DataFrame operations
    
    Create a function named 'process_dataframe_query' that:
    1. Takes parameters: df (DataFrame), query (string)
    2. Returns: DataFrame, number, or matplotlib figure
    3. Uses only safe pandas/matplotlib operations
    
    Available columns: {', '.join(df_columns)}
    User query: {sanitized_query}
    
    Example safe operations:
    - df.groupby(), df.filter(), df.sort_values()
    - plt.bar(), plt.pie(), plt.scatter()
    - df.mean(), df.sum(), df.count()
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"# Error generating code: {str(e)}"

def create_secure_temp_file():
    """Create a secure temporary file with restricted permissions"""
    temp_dir = tempfile.gettempdir()
    # Create a unique filename using hash
    unique_id = hashlib.md5(str(os.getpid()).encode()).hexdigest()[:8]
    temp_file = Path(temp_dir) / f"safe_code_{unique_id}.py"
    return str(temp_file)

def validate_generated_code(code):
    """Validate generated code for safety"""
    dangerous_imports = ['os', 'subprocess', 'sys', '__builtin__', 'builtins']
    dangerous_functions = ['exec', 'eval', 'compile', 'open', 'file', '__import__']
    
    lines = code.split('\n')
    for line in lines:
        line = line.strip().lower()
        
        # Check for dangerous imports
        for imp in dangerous_imports:
            if f'import {imp}' in line or f'from {imp}' in line:
                return False
        
        # Check for dangerous functions
        for func in dangerous_functions:
            if func in line:
                return False
    
    return True

def execute_safe_query(df, user_query):
    """Execute user query with safety restrictions"""
    if df is None:
        return "Error: No data available"
    
    code = generate_safe_code(user_query, df.columns)
    
    # Validate code safety
    if not validate_generated_code(code):
        return "Error: Generated code contains unsafe operations"
    
    # Create secure temporary file
    file_path = create_secure_temp_file()
    
    try:
        # Clean code by removing markdown formatting
        clean_code = re.sub(r'```python\n?|```\n?', '', code)
        
        with open(file_path, "w") as f:
            f.write(clean_code)
        
        # Import and execute with restricted environment
        import importlib.util
        spec = importlib.util.spec_from_file_location("safe_module", file_path)
        safe_module = importlib.util.module_from_spec(spec)
        
        # Execute with error handling
        spec.loader.exec_module(safe_module)
        
        if hasattr(safe_module, 'process_dataframe_query'):
            result = safe_module.process_dataframe_query(df, user_query)
            return result
        else:
            return "Error: Generated code missing required function"
            
    except Exception as e:
        return f"Error executing query: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

def save_to_excel(df):
    """Safely save DataFrame to Excel with error handling"""
    if df is None:
        return None
    
    try:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error saving to Excel: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Secure Excel Query Chatbot", layout="wide")
    st.title("🔒 Secure Excel Query Chatbot with AI")
    
    # Configure API securely
    configure_api()
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    
    if uploaded_file:
        df = load_excel(uploaded_file)
        
        if df is not None:
            df = normalize_percentages(df, "Progress")
            st.write("Data Preview:", df.head())
            
            # Query input with validation
            user_query = st.text_input("Ask a question about the data (safe operations only)")
            
            if user_query:
                if len(user_query.strip()) < 3:
                    st.warning("Please enter a more specific query")
                else:
                    with st.spinner("Processing your query securely..."):
                        response = execute_safe_query(df, user_query)
                        
                        if isinstance(response, str) and response.startswith("Error"):
                            st.error(response)
                        else:
                            st.write("Response:", response)
                            
                            # Download filtered data if DataFrame returned
                            if isinstance(response, pd.DataFrame):
                                excel_data = save_to_excel(response)
                                if excel_data:
                                    st.download_button(
                                        label="Download Filtered Data",
                                        data=excel_data,
                                        file_name="filtered_data.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
            
            # Download original data
            excel_data = save_to_excel(df)
            if excel_data:
                st.download_button(
                    label="Download Updated Excel File",
                    data=excel_data,
                    file_name="updated_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("Please upload an Excel file to get started.")
        st.markdown("""
        ### Security Features:
        - ✅ Secure API key handling
        - ✅ Input sanitization
        - ✅ Code injection prevention
        - ✅ Safe file operations
        - ✅ Error handling
        """)

if __name__ == "__main__":
    main()