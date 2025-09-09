from flask import Flask, render_template, request, send_file
import pandas as pd
from io import BytesIO
import os
import google.generativeai as genai
from dotenv import load_dotenv
import tempfile
import importlib
import subprocess
import sys
import matplotlib.pyplot as plt

# Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyDBdBlw2WkCrINQJV5oZ-Vkq8dLOLd5Nls")

# Function to load Excel data into a DataFrame
def load_excel(file):
    df = pd.read_excel(file)
    return df

# Function to normalize percentages in a specified column
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

1. Be named `process_dataframe_query`.
2. Accept two arguments:
   - `df`: A Pandas DataFrame containing the data.
   - `query`: A string containing the user's query.
3. Perform operations or visualizations as described in the query.
4. Return one of the following based on the query:
   - A new Pandas DataFrame (e.g., after filtering, sorting, or grouping).
   - A summary statistic (e.g., total, mean, or count).
   - A Matplotlib figure for visualizations (e.g., bar chart, pie chart, or scatter plot).

**DataFrame Schema**:
- Assume the DataFrame has the following columns: {', '.join(df_columns)}.

User Query: {user_query}
    """
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    code = response.text.strip()
    return code

def execute_code_query(df, user_query):
    code = generate_python_code(user_query, df.columns)
    file_path = "generated_code.py"
    try:
        with open(file_path, "w") as f:
            f.write(code)
        delete_first_last_lines(file_path)

        spec = importlib.util.spec_from_file_location("generated_module", file_path)
        generated_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generated_module)

        result = generated_module.process_dataframe_query(df, user_query)
        if isinstance(result, plt.Figure):
            img_path = "static/query_result.png"
            result.savefig(img_path)
            plt.close(result)
            return {"type": "visual", "path": img_path}
        elif isinstance(result, pd.DataFrame):
            output = BytesIO()
            result.to_excel(output, index=False)
            output.seek(0)
            return {"type": "dataframe", "content": output}
        else:
            return {"type": "text", "content": result}
    except Exception as e:
        return {"type": "error", "content": str(e)}

@app.route("/", methods=["GET", "POST"])
def home():
    # Load the data file
    data_path = "data/datafile.xlsx"
    if not os.path.exists(data_path):
        return "Data file not found. Please ensure 'datafile.xlsx' exists in the 'data' folder."

    df = load_excel(data_path)
    df = normalize_percentages(df, "Progress")

    if request.method == "POST":
        user_query = request.form.get("query")
        result = execute_code_query(df, user_query)

        if result["type"] == "visual":
            return render_template("chatdesign.html", img_path=result["path"])
        elif result["type"] == "dataframe":
            return send_file(result["content"], as_attachment=True, download_name="filtered_data.xlsx")
        elif result["type"] == "text":
            return render_template("chatdesign.html", response=result["content"])
        else:
            return render_template("chatdesign.html", response="An error occurred.")

    return render_template("chatdesign.html")

if __name__ == "__main__":
    app.run(debug=True)
