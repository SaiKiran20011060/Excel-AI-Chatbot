import pandas as pd
import matplotlib.pyplot as plt

def process_dataframe_query(df, query):
    """
    Processes a Pandas DataFrame based on a natural language query.

    Args:
        df: A Pandas DataFrame.
        query: A string containing the user's query.

    Returns:
        A Pandas DataFrame, a summary statistic, a Matplotlib figure, or None 
        if the query is not understood.
    """

    query = query.lower()

    if "pie chart of progress" in query:
        # Check if 'Progress (%)' column exists
        if 'Progress (%)' not in df.columns:
            return "Error: 'Progress (%)' column not found in DataFrame."

        progress_counts = df['Progress (%)'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(progress_counts, labels=progress_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('Progress Distribution')
        return fig

    elif "average progress" in query:
        if 'Progress (%)' not in df.columns:
            return "Error: 'Progress (%)' column not found in DataFrame."
        return df['Progress (%)'].mean()

    elif "filter by status" in query and "status" in query:
        status = query.split("status ")[-1].strip()  #Extract status from query
        #Simple status check, can be improved with regex for complex status queries
        filtered_df = df[df['Status'] == status]
        return filtered_df
    
    elif "show tasks due before" in query:
        try:
          date_str = query.split("before")[-1].strip()
          #Basic date parsing, needs more robust solution for diverse date formats
          due_date = pd.to_datetime(date_str)
          filtered_df = df[pd.to_datetime(df['Due Date']) < due_date]
          return filtered_df
        except (ValueError, KeyError):
          return "Error: Invalid date format or 'Due Date' column not found."


    else:
        return "Query not understood."



# Example usage:
data = {'Task': ['A', 'B', 'C', 'D'],
        'Progress (%)': [25, 50, 75, 100],
        'Status': ['In Progress', 'In Progress', 'Completed', 'Completed'],
        'Due Date': ['2024-03-15', '2024-03-22', '2024-03-10', '2024-03-29']}
df = pd.DataFrame(data)

query1 = "show me the pie chart of progress"
result1 = process_dataframe_query(df, query1)
if isinstance(result1, plt.Figure):
    result1.show()


query2 = "average progress"
result2 = process_dataframe_query(df, query2)
print(f"Average Progress: {result2}")

query3 = "filter by status completed"
result3 = process_dataframe_query(df, query3)
print(f"Tasks with status 'Completed':\n{result3}")

query4 = "show tasks due before 2024-03-18"
result4 = process_dataframe_query(df, query4)
print(f"Tasks due before 2024-03-18:\n{result4}")

query5 = "invalid query"
result5 = process_dataframe_query(df, query5)
print(f"Result for invalid query: {result5}")

