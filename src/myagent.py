"""
 My Idea of an AI Agent for the LLama2 llm.

 Separate tools for:
  - Average
  - Mean
  - Median
  - Sum
  - Div
  - Diff
  - Max
  - Min
  etc

  LLm gets prompted with the user question + the whole chart info JSON Object + available tools
  LLM should decide what tool to use, but only for calculations purposes.
  LLm should not only decide what tool to use, but also use the tool description and parameter descriptions to reason about what exact values does it need to calculate.
  Ideas:
   * A loop to use tools as many as needed to answer the user prompt
   * A loop if something goes wrong - Error to the LLm and retry'
   * Do not return a single tool, return a list of tools/parameters pair to answer more complex questions
"""

from typing import List, Dict, Any, Optional, Union
import json
import re
from src.llm import build_llm_for_chart_drawing
from langchain.llms import Ollama

# Tool implementations
def calculate_average(values: List[float]) -> float:
    """Calculate the arithmetic mean of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def calculate_median(values: List[float]) -> float:
    """Calculate the median of a list of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]


def calculate_sum(values: List[float]) -> float:
    """Calculate the sum of a list of values."""
    return sum(values)


def calculate_max(values: List[float]) -> float:
    """Calculate the maximum value from a list."""
    if not values:
        return 0.0
    return max(values)


def calculate_min(values: List[float]) -> float:
    """Calculate the minimum value from a list."""
    if not values:
        return 0.0
    return min(values)


def extract_values_from_chart_data(chart_data: Dict[str, Any], target: str = "all") -> List[float]:
    """Extract numerical values from chart data."""
    values = []

    # Try to extract from content if it contains numerical data
    content = chart_data.get('content', '')
    data_table = chart_data.get('data_table', '')

    # Look for numbers in the content
    number_pattern = r'\b\d+\.?\d*\b'
    numbers = re.findall(number_pattern, content + ' ' + data_table)

    try:
        values = [float(num) for num in numbers if float(num) > 0]
    except:
        values = []

    return values


def run_chart_analysis(question: str, chart_data: Dict[str, Any]) -> str:
    """Run the chart analysis with simplified tool approach for local LLM."""

    # Create a prompt that asks the LLM to analyze and use tools
    prompt = f"""You are a data analyst. Analyze the chart data and answer the question using the available calculation tools.

CHART DATA:
{json.dumps(chart_data, indent=2)}

QUESTION: {question}

AVAILABLE TOOLS:
1. extract_values_from_chart_data(chart_data, target="all") - Extract numerical values from chart
2. calculate_average(values) - Calculate average of values  
3. calculate_sum(values) - Calculate sum of values
4. calculate_median(values) - Calculate median of values
5. calculate_max(values) - Calculate maximum value
6. calculate_min(values) - Calculate minimum value

INSTRUCTIONS:
1. First, extract relevant values from the chart data
2. Then apply appropriate calculations
3. Provide a clear answer with your reasoning

Use this format for tool calls:
TOOL[tool_name(parameters)]

Example:
TOOL[extract_values_from_chart_data(chart_data, "all")]
TOOL[calculate_average([4.3, 2.44, 2.0, 3.02])]

Your analysis:
"""

    try:
        llm = Ollama(model="llama3.2")
        response = llm.invoke(prompt)

        # Process tool calls in the response
        processed_response = process_tool_calls(response, chart_data)

        return processed_response

    except Exception as e:
        return f"Error in chart analysis: {str(e)}"


def process_tool_calls(response: str, chart_data: Dict[str, Any]) -> str:
    """Process TOOL[...] calls in the LLM response."""

    # Find all TOOL[...] patterns
    tool_pattern = r'TOOL\[(.*?)\]'
    matches = re.findall(tool_pattern, response)

    processed_response = response

    for match in matches:
        try:
            # Execute the tool call
            result = execute_tool_call(match, chart_data)

            # Replace the TOOL[...] with the actual result
            processed_response = processed_response.replace(f'TOOL[{match}]', str(result))

        except Exception as e:
            # Replace with error message if tool fails
            processed_response = processed_response.replace(f'TOOL[{match}]', f'[Error: {str(e)}]')

    return processed_response


def execute_tool_call(tool_call: str, chart_data: Dict[str, Any]) -> Union[float, List[float]]:
    """Execute a tool call safely."""

    # Create a safe execution environment with available tools
    safe_globals = {
        'calculate_average': calculate_average,
        'calculate_median': calculate_median,
        'calculate_sum': calculate_sum,
        'calculate_max': calculate_max,
        'calculate_min': calculate_min,
        'extract_values_from_chart_data': extract_values_from_chart_data,
        'chart_data': chart_data
    }

    try:
        # Execute the tool call safely
        result = eval(tool_call, {"__builtins__": {}}, safe_globals)

        # Format result appropriately
        if isinstance(result, float):
            return round(result, 2)
        elif isinstance(result, list):
            return result
        else:
            return result

    except Exception as e:
        raise Exception(f"Tool execution failed: {str(e)}")
