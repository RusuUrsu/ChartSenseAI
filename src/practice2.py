import ast

from langchain_community.llms import Ollama
from langchain.agents import Tool, initialize_agent, AgentType
import json
from typing import List

def get_charts_from_json(json_path):
    with open(json_path, "r", encoding='utf-8') as json_file:
        chart = json.load(json_file)

    return chart

"""
select tool -> select label -> call tool -> ...
"""

chart_info = {
    "data_table": [
      "TITLE | % of Students Enrolled",
      "Humanities | 22% ",
      " Business | 30% ",
      " Education | 13% ",
      " Engineering | 20% ",
      " Architecture | 15%"
    ],
    "file_path": "extracted_images\\page_1_image_1.jpeg",
    "chart_type": "Pie Chart",
    "nearby_text": "Here we have a Pie Chart that shows the percentage of students enrolled in each m Humanities, Business, Education, Engineering and Architecture.",
    "chart_title": "% of Students Enrolled",
    "labels": [
      "Humanities",
      "Business",
      "Education",
      "Engineering",
      "Architecture"
    ],
    "values": [
      22.0,
      30.0,
      13.0,
      20.0,
      15.0
    ],
    "page_number": 1,
    "image_number": 1
}
import logging
logger = logging.getLogger(__name__)

def calculate_average(label: str) -> float:
    values = extract_values(label)
    try:
        ans = sum(values) / len(values)
        return ans
    except Exception as e:
        logger.info(e)

def calculate_sum(label: str) -> float:
    values = extract_values(label)
    try:
        ans = sum(values)
        return ans
    except Exception as e:
        logger.info(e)
        return 0

def calculate_max(label: str) -> float:
    values = extract_values(label)
    try:
        ans = max(values)
        return ans
    except Exception as e:
        logger.info(e)
        return 0

def calculate_min(label: str) -> float:
    values = extract_values(label)
    try:
        ans = min(values)
        return ans
    except Exception as e:
        logger.info(e)



def extract_values(label: str):
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

def compute_statistics(labels: List[str]):
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
def run_agent(question: str, image_number: int=2):
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

def get_labels():
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


def create_chart_prompt(chart_info, question_text):

    try:
        # Create a formatted data representation based on chart type
        if chart_info['chart_type'] == "Bar Chart":
            data_representation = "Data values by category:\n"
            for category in chart_info['categories']:
                values_str = ", ".join([str(v) if v is not None else "N/A" for v in chart_info['data'][category]])
                data_representation += f"- {category}: {values_str} for {', '.join(chart_info['x_labels'])} respectively\n"
        elif chart_info['chart_type'] == "Pie Chart":
            data_representation = "Pie chart values:\n"
            for i, label in enumerate(chart_info['labels']):
                data_representation += f"- {label}: {chart_info['values'][i]}\n"
        # Other chart type formatting remains the same
        else:
            data_representation = f"Raw data: {chart_info['data_table']}"

        # Build the complete prompt with improved guidance
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

#print(calculate_sum("Audi"))