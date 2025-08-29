'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formatting of spacing and indentation of the prompt template is important for Llama-2-7B-Chat,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly

qa_template = """
Use the following pieces of context to answer the question at the end.
First, analyze the information provided and the user question. 
Then, reason step-by-step about what the answer could be.
Finally, provide your conclusion based on this reasoning.

Context: {context}
Question: {question}

Include the reason for your response.
Do not mention all reasoning steps in your response.
Your Answer:
"""

agentic_qa_template = """You are an intelligent data analyst with mathematical reasoning capabilities.

CHART/TABLE DATA:
{context}

{calculations}

USER QUESTION: {question}

INSTRUCTIONS:
- Analyze the data carefully and use any provided calculations
- If mathematical operations were performed, reference them in your answer
- Be precise with numbers and provide specific values when possible
- For questions about trends, comparisons, or rankings, use the data to give exact answers
- Show your reasoning step by step

ANSWER:
"""

parsing_template = """
You will receive as input the type of the chart and the table data of this chart.
Convert this table string into a structured JSON object with chart_type, x, y, series (if multiple), values
chart_type = {chart_type}
table_data = {table_data}

Example Input:
Chart type: Line Chart
Table Data: Month | Sales A | Sales B 
Jan | 100 | 120 
Feb | 130 | 140 
Mar | 150 | 160

Example Output:
{{
  "chart_type": "Line Chart",
  "x": ["Jan", "Feb", "Mar"]
  "series": {
    "Sales A": [100, 130, 150],
    "Sales B": [120, 140, 160]
  }
}}

Respond only with the JSON object.
Your answer:
"""

reconstruction_template = """
Given this string: 
{table_data}
 Knowing that this is a {chart_type} chart extract me the labels (if any), values and all other data necessary for reconstructing the chart in a JSON object
 If chart type is unknown, reason over what chart type it could be and add a new attribute in the JSON object for it.

Response:
"""

mathematical_analysis_template = """
You are a mathematical analyst examining chart data. Perform the following analysis:

DATA: {table_data}
ANALYSIS TYPE: {analysis_type}

Available mathematical operations:
- Calculate averages, means, medians
- Find maximum and minimum values
- Determine trends and patterns
- Calculate sums and totals
- Compute standard deviations and variance

Based on the data and analysis type, provide:
1. Relevant calculations
2. Key insights
3. Patterns or trends observed
4. Specific numerical findings

Analysis:
"""

agent_prompt = """You are an advanced AI data analyst. Your task is to answer a question about a chart by using a set of available tools.

Here are the tools you can use:
{tools}

You will be given a question and the context of the chart, including its title, type, and the data table.

**Instructions:**
1.  Analyze the user's question and the provided chart context.
2.  Determine if any of the available tools can help you answer the question more accurately.
3.  If a tool is suitable, respond with a JSON object containing the tool's name and the input data extracted from the chart's data table. The JSON should be in the following format:
    ```json
    {{
      "tool_name": "name_of_the_tool",
      "tool_input": {{
        "values": [extracted_data]
      }}
    }}
    ```
4.  If no tool is necessary, or if you cannot extract the required data, simply answer the question based on the provided context.

**Chart Context:**
{context}

**User Question:**
{question}

Your response:
"""
