# Secure Excel Query Chatbot

A secure Streamlit application that allows users to query Excel data using natural language with AI assistance.

## Security Features

- ✅ Secure API key handling via environment variables
- ✅ Input sanitization to prevent code injection
- ✅ Restricted code execution environment
- ✅ Safe file operations with temporary file cleanup
- ✅ Comprehensive error handling

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API Key**
   - Copy `.env.example` to `.env`
   - Add your Google API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_actual_api_key_here
     ```

3. **Run the Application**
   ```bash
   streamlit run ExcelChat_Secure.py
   ```

## Usage

1. Upload an Excel file (.xlsx format)
2. Ask natural language questions about your data
3. Get AI-generated insights, visualizations, or filtered data
4. Download results as Excel files

## Supported Query Types

- **Data Filtering**: "Show me records where progress is less than 50%"
- **Visualizations**: "Create a pie chart of the progress column"
- **Statistics**: "What's the average progress across all projects?"
- **Sorting**: "Sort the data by progress in descending order"

## Security Considerations

- The application sanitizes all user inputs
- Generated code is validated before execution
- Only safe pandas and matplotlib operations are allowed
- No system calls or file operations in generated code
- API keys are never hardcoded in the source code

## File Structure

```
streamlitchat/
├── ExcelChat.py              # Original version (has security issues)
├── ExcelChat_Secure.py       # Secure version (recommended)
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variable template
└── README.md               # This file
```

## Migration from Original Version

If you're using the original `ExcelChat.py`, please migrate to `ExcelChat_Secure.py` for better security:

1. Remove the hardcoded API key from your code
2. Set up environment variables as described above
3. Use the secure version for production deployments