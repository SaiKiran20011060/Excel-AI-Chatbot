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

genai.configure(api_key="AIzaSyDBdBlw2WkCrINQJV5oZ-Vkq8dLOLd5Nls")

def load_excel(file):
    df = pd.read_excel(file) 
    return df

def normalize_percentages(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name] * 100
    return df

def delete_first_last_lines(filepath):
    """Deletes the first and last lines of a file."""
    try:
        with open(filepath, 'r') as f_in:
            lines = f_in.readlines()

        if len(lines) > 2:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f_out:
                f_out.writelines(lines[1:-1])
            os.replace(f_out.name, filepath)
        elif len(lines) > 0:
            with open(filepath, 'w') as f:
                f.write("")  # Clears the file

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_python_code(user_query, df_columns):
    prompt = f"""
        You are a Python data analysis and visualization expert. Your task is to generate Python code that processes a Pandas DataFrame based on a user's natural language query. The generated function should:

1. Be named process_dataframe_query.
2. Accept two arguments:
   - df: A Pandas DataFrame containing the data.
   - query: A string containing the user's query.
3. Perform operations or visualizations as described in the query.
4. Return one of the following based on the query:
   - A new Pandas DataFrame (e.g., after filtering, sorting, or grouping).
   - A summary statistic (e.g., total, mean, or count).
   - A Matplotlib figure for visualizations (e.g., bar chart, pie chart, or scatter plot).

**Guidelines**:
- For visualization queries, create the appropriate chart and return the Matplotlib figure object.
- For data transformations (e.g., filtering or sorting), return a new DataFrame.
- If the query is unclear or unsupported, return an error message as a string.

**Examples**:
1. Query: "Show me the progress in a pie chart"
   - Output: A pie chart visualizing the "Progress" column's percentage distribution.

2. Query: "Filter tasks with progress less than 50% and show them"
   - Output: A DataFrame filtered where "Progress" is less than 50%.

3. Query: "What is the average progress across all projects?"
   - Output: A number representing the average progress.

**DataFrame Schema**:
- Assume the DataFrame has the following columns: {', '.join(df_columns)}. These column names are case-sensitive.

User Query: {user_query}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    code = response.text.strip()
    return code

def execute_code_query(df, user_query):
    code = generate_python_code(user_query, df.columns)
    
    # Save the generated code to a temporary file
    file_path = "generated_code.py"
    try:
        with open(file_path, "w") as f:
            f.write(code)
        delete_first_last_lines(file_path)

        # Dynamically import the generated code
        spec = importlib.util.spec_from_file_location("generated_module", file_path)
        generated_module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(generated_module)
        except ModuleNotFoundError as e:
            missing_module = str(e).split("'")[1]
            subprocess.check_call([sys.executable, "-m", "pip", "install", missing_module])
            spec.loader.exec_module(generated_module)

        # Execute the function in the generated module
        result = generated_module.process_dataframe_query(df, user_query)

        # Handle visualizations or DataFrame responses
        # if isinstance(result, plt.Figure):
        #     st.pyplot(result)
        #     plt.close(result)
        # elif isinstance(result, pd.DataFrame):
        #     st.write(result)  # Display filtered or processed DataFrame
        # else:
        #     st.write(result)  # Display other outputs like text or summaries

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
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:  # Ensure xlsxwriter is used
        df.to_excel(writer, index=False)
    output.seek(0)
    return output

# Main Streamlit application
def main():
    st.set_page_config(page_title="Excel Query Chatbot with AI", layout="wide")
    st.title("Excel Query Chatbot with AI")

    # Hardcoded file path to load data
    file_path = r"C:\Users\saiki\OneDrive\Desktop\SAI\Projects\Excel_Chat\streamlitchat\datafile.xlsx"  # Update this path with your Excel file
    if os.path.exists(file_path):
        df = load_excel(file_path)
        df = normalize_percentages(df, "Progress")
        st.write("Data Preview:", df.head(0))

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
        st.error(f"File '{file_path}' not found. Please ensure the file exists.")

if __name__ == "__main__":
    main()