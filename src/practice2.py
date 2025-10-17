"""
Chart Question Answering Agent Module

This module implements an agentic approach to answer questions about chart data
using ReAct (Reasoning + Acting) pattern. It provides tools for statistical
calculations on chart data and uses an LLM to determine which calculations are needed.

- Tools: Agent-accessible functions for chart calculations
- LLM: Ollama Mistral model for reasoning and decision-making
- Agent Type: ZERO_SHOT_REACT_DESCRIPTION for flexible problem-solving
- Prompt Engineering: Dynamic prompts tailored to chart structure
"""

import ast
from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
import json
from typing import List
import logging

def get_charts_from_json(json_path: str) -> list:
    """
    Load chart data from the JSON file. (created in Extraction Module)
    Args:
        json_path (str): Path to the JSON file containing chart metadata.
        
    Returns:
        list: List of chart dictionaries from the JSON file.
    """
    with open(json_path, "r", encoding='utf-8') as json_file:
        chart = json.load(json_file)
    return chart


logger = logging.getLogger(__name__)

def calculate_average(label: str) -> float:
    """
    Calculate the average value for a chart label or category.
    
    Extracts numeric values associated with a label and computes their mean.
    
    Args:
        label (str): The label or category name to calculate average for.
        
    Returns:
        float: The average of the values, or None if an error occurs.
    """
    values = extract_values(label)
    try:
        ans = sum(values) / len(values)
        return ans
    except Exception as e:
        logger.info(e)

def calculate_sum(label: str) -> float:
    """
    Calculate the sum of values for a chart label or category.
    
    Extracts numeric values associated with a label and computes their total.
    
    Args:
        label (str): The label or category name to sum values for.
        
    Returns:
        float: The sum of the values. Returns 0 if an error occurs.
    """
    values = extract_values(label)
    try:
        ans = sum(values)
        return ans
    except Exception as e:
        logger.info(e)
        return 0

def calculate_max(label: str) -> float:
    """
    Find the maximum value for a chart label or category.
    
    Extracts numeric values associated with a label and finds the maximum.
    
    Args:
        label (str): The label or category name to find maximum for.
        
    Returns:
        float: The maximum value. Returns 0 if an error occurs.
    """
    values = extract_values(label)
    try:
        ans = max(values)
        return ans
    except Exception as e:
        logger.info(e)
        return 0

def calculate_min(label: str) -> float:
    """
    Find the minimum value for a chart label or category.
    
    Extracts numeric values associated with a label and finds the minimum.
    
    Args:
        label (str): The label or category name to find minimum for.
        
    Returns:
        float: The minimum value. Returns None if an error occurs.
    """
    values = extract_values(label)
    try:
        ans = min(values)
        return ans
    except Exception as e:
        logger.info(e)


def extract_values(label: str) -> list:
    """
    Extract numeric values from chart data for a given label.
    
    Handles different chart types and extracts values based on the label.
    Cleans label strings and handles both category and axis labels.
    
    Args:
        label (str): The label to extract values for. Can be:
                    - Category name (for Bar/Line charts)
                    - Axis label (x, y, x_label, y_label for Scatter plots)
                    - Pie chart labels (returns all values)
        
    Returns:
        list: List of numeric values. Empty list if extraction fails or no values found.
        
    Supported Chart Types:
        - Bar Chart: Extract from categories or x_labels
        - Pie Chart: Returns all values
        - Scatter Plot: Extract x or y values
        - Line Chart: Extract from series or x_values
        - can be extended to support more chart types
    """
    label = label.strip("'\"")
    if isinstance(label, list):
        label = label[0].strip("'\"")
    try:
        if chart_info['chart_type'] == "Bar Chart":
            if label in chart_info['categories']:
                return [v for v in chart_info['data'][label] if v is not None]
            elif label in chart_info['x_labels']:
                idx = chart_info['x_labels'].index(label)
                return [chart_info['data'][cat][idx] for cat in chart_info['categories'] if chart_info['data'][cat][idx] is not None]
        elif chart_info['chart_type'] == "Pie Chart":
            return chart_info['values']
        elif chart_info['chart_type'] == "Scatter Plot":
            if label.lower() == 'x' or label == chart_info['x_label']:
                return chart_info['x_values']
            elif label.lower() == 'y' or label == chart_info['y_label']:
                return chart_info['y_values']
        elif chart_info['chart_type'] == "Line Chart":
            if label in chart_info['categories']:
                return [v for v in chart_info['series'][label] if v is not None]
            elif label in chart_info['x_values']:
                idx = chart_info['x_values'].index(label)
                return [chart_info['series'][cat][idx] for cat in chart_info['categories'] if chart_info['series'][cat][idx] is not None]
        else:
            raise Exception(f"Unknown chart type: {chart_info['chart_type']}")
    except Exception as e:
            logger.info(e)
            return []

def compute_statistics(labels: List[str]) -> dict:
    """
    Compute all statistics (average, sum, max, min) for multiple labels.
    
    Takes a list of labels and calculates all statistical measures for each,
    providing an overview of the chart data.
    
    Args:
        labels (List[str]): List of label names as a string representation of a list.
                           E.g., "['label1', 'label2']"
        
    Returns:
        dict: Dictionary mapping labels to their statistics:
              {
                  "label1": {
                      "average": float,
                      "sum": float,
                      "max": float,
                      "min": float
                  },
                  ...
              }
    """
    stats = {}
    labels = ast.literal_eval(labels)
    for label in labels:
        values = extract_values(label)
        if values:
            stats[label] = {
                "average": sum(values) / len(values),
                "sum": sum(values),
                "max": max(values),
                "min": min(values)
            }
    return stats

tools = [
    Tool(
        name="calculate_average",
        func=calculate_average,
        description="Always use to compute the average for a column or row in the chart data. Input should be the exact label name."
    ),
    Tool(
        name="calculate_sum",
        func=calculate_sum,
        description="Compute the sum for a column or row in the chart data. Input should be the exact label name."
    ),
    Tool(
        name="calculate_max",
        func=calculate_max,
        description="Compute the maximum value for a column or row in the chart data. Input should be the exact label name."
    ),
    Tool(
        name="calculate_min",
        func=calculate_min,
        description="Compute the minimum value for a column or row in the chart data. Input should be a string representing the exact label name."
    ),
    Tool(
        name="compute_statistics",
        func=compute_statistics,
        description="Compute all statistics (average, sum, max, min) for all available labels in the chart data. Helps if user asks for overview of the chart. Input should be a list of all available labels."
    )
]

def run_agent(question: str, image_number: int = 2) -> str:
    """
    Run the ReAct agent to answer a question about a specific chart.
    
    Initializes the agent with tools for statistical calculations, loads the
    chart data, creates a contextual prompt, and uses the LLM to reason about
    and answer the question.
    - Uses Ollama Mistral model with temperature=0.1 (slightly low) for more deterministic responses
    - Supports up to 10 reasoning iterations
    - Handles parsing errors gracefully
    
    Args:
        question (str): The question to ask about the chart.
        image_number (int): Index (1-based) of the chart to analyze from 
                           extracted_images/extraction_results.json.
                           Defaults to 2.
        
    Returns:
        str: The agent's answer to the question.
       
    """
    global chart_info
    chart_info = get_charts_from_json("extracted_images/extraction_results.json")[image_number-1]
    agent = initialize_agent(
        tools=tools,
        llm=Ollama(model='mistral', temperature=0.1),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True
    )
    result = agent.run(create_chart_prompt(chart_info,question))
    return result

def get_labels() -> list:
    """
    Get all available labels and categories from the current chart.
    
    Returns the appropriate labels based on the chart type.
    
    Returns:
        list: List of label/category names:
              - Bar Chart: categories + x_labels
              - Pie Chart: categories
              - Scatter Plot: [x_label, y_label, 'x', 'y']
              - Line Chart: categories + x_values
    """
    labels = []
    if chart_info['chart_type'] == "Bar Chart":
        labels = chart_info['categories'] + chart_info['x_labels']
    elif chart_info['chart_type'] == "Pie Chart":
        labels = chart_info['categories']
    elif chart_info['chart_type'] == "Scatter Plot":
        labels = [chart_info['x_label'], chart_info['y_label'], 'x', 'y']
    elif chart_info['chart_type'] == "Line Chart":
        labels = chart_info['categories'] + chart_info['x_values']
    return labels


def create_chart_prompt(chart_info: dict, question_text: str) -> str:
    """
    Create a contextual prompt for the LLM agent.
    
    Builds a detailed prompt that includes chart information, data representation,
    decision process guidance, and the user's question. Supports different
    chart types with appropriate data formatting.
    
    Args:
        chart_info (dict): Dictionary containing chart metadata:
                          - chart_type: Type of chart (Bar, Pie, Scatter, Line)
                          - chart_title: Title of the chart
                          - categories/labels: Chart categories
                          - data/values/series: Chart data values
                          - x_labels/x_values: X-axis labels/values
                          - data_table: Raw table data
        question_text (str): The user's question about the chart.
        
    Returns:
        str: A formatted prompt string with:
             1. Chart description
             2. Data representation (formatted by chart type)
             3. Decision process guidance
             4. Available tools
             5. The question and response format instructions
             
    """
    try:
        if chart_info['chart_type'] == "Bar Chart":
            data_representation = "Data values by category:\n"
            for category in chart_info['categories']:
                values_str = ", ".join([str(v) if v is not None else "N/A" for v in chart_info['data'][category]])
                data_representation += f"- {category}: {values_str} for {', '.join(chart_info['x_labels'])} respectively\n"
        elif chart_info['chart_type'] == "Pie Chart":
            data_representation = "Pie chart values:\n"
            for i, label in enumerate(chart_info['labels']):
                data_representation += f"- {label}: {chart_info['values'][i]}\n"
        else:
            data_representation = f"Raw data: {chart_info['data_table']}"

        prompt = f"""
    I have a {chart_info['chart_type']} titled "{chart_info.get('chart_title', 'Chart')}"
    
    Context information:
    - Labels/Categories: {', '.join(chart_info.get('x_labels', []))} + {', '.join(chart_info.get('categories', []))}
    - Data Table: {data_representation}
    
    DECISION PROCESS:
    1. First, determine if the question requires calculations or can be answered directly from the data
    2. If calculations are needed, select and use the appropriate tool
    3. If tools fail or aren't needed, answer directly using the chart data provided
    4. Use both the tool observations and the context to formulate your final answer
    
    {question_text}
    
    IMPORTANT INSTRUCTIONS:
    - If a tool is needed:
      - Write exactly:
        Thought: I need to calculate something.
        Action: <tool_name>
        Action Input: "<exact label>" (from the list of labels/categories)
    - If you can give the final answer:
        - Write exactly:
            Thought: I know the final answer
            Final Answer: <your answer>
            
    Available tools: calculate_average, calculate_sum, calculate_max, calculate_min, compute_statistics
    
    Question: {question_text}
    Your complete answer:
    """
        return prompt
    except Exception as e:
        return f"""
        You are given a chart and its data table. Knowing that the chart type is {chart_info['chart_type']},
        {chart_info['data_table']}
        
        Answer the question: {question_text}
        If you do not know the answer, simply refuse to respond.
        Your answer:
        """