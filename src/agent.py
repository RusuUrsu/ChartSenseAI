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


# ----------------------------

def compute_statistics_labels(labels: List[str]):
    """Compute average, sum, max, min for given chart labels."""
    stats = {}
    try:
        labels = ast.literal_eval(labels)
    except Exception:
        logger.error(f"Failed to parse labels: {labels}")
        return {}

    for label in labels:
        label = label.strip("'\"")
        values = extract_values(label)
        if values:
            stats[label] = {
                "average": sum(values) / len(values),
                "sum": sum(values),
                "max": max(values),
                "min": min(values)
            }
    return stats


def compute_statistics_numbers(numbers: List[float]):
    """Compute average, sum, max, min for a raw list of numbers."""
    try:
        numbers = ast.literal_eval(numbers)
        numbers = [float(n) for n in numbers]
    except Exception:
        logger.error(f"Failed to parse numbers: {numbers}")
        return {}

    if not numbers:
        return {}

    return {
        "average": sum(numbers) / len(numbers),
        "sum": sum(numbers),
        "max": max(numbers),
        "min": min(numbers)
    }

tools = [
    Tool(
        name="compute_statistics_numbers",
        func=compute_statistics_numbers,
        description="Compute statistics (average, sum, max, min) for a list of numerical values."
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
    if chart_info.get('chart_type') == "Pie Chart":
        data_repr = "\n".join([f"- {l}: {v}" for l, v in zip(chart_info.get('labels', []), chart_info.get('values', []))])
    else:
        data_repr = f"Raw data:\n{chart_info.get('data_table', [])}"

    return f"""
You are analyzing a {chart_info.get('chart_type', 'Chart')} titled "{chart_info.get('chart_title', 'Chart')}".
Chart data:
{data_repr}

### Workflow Rules:
1. Think step-by-step to solve the question.
3. Use a tool if calculations are required; otherwise answer directly.
4. Provide a final answer when done.

### Tools Available:
- compute_statistics_numbers: for a list of numerical values.

Answer Format:
- If you can answer directly:
Thought: I can answer this directly from the chart.
Final Answer: <your answer>

- If a tool is needed:
Thought: I need to calculate something.
Action: <tool_name>
Action Input: <list of numbers> (e.g., [10, 20, 30]) Extract numbers from the chart data or from earlier tool outputs.
Question: {question_text}
Your answer:
"""


#print(calculate_sum("Audi"))