import pandas as pd

# Create a sample DataFrame
data = {
    "Task": ["Design Website", "Write Content", "Develop Backend", "Testing", "Launch Product"],
    "Progress (%)": [75, 50, 30, 90, 100],
    "Status": ["Ongoing", "Ongoing", "Pending", "Ongoing", "Complete"],
    "Due Date": ["2024-01-15", "2024-01-20", "2024-02-01", "2024-01-10", "2024-01-25"]
}

df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
file_path = "datafile.xlsx"
df.to_excel(file_path, index=False)

file_path
