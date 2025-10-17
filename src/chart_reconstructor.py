"""
Chart Reconstructor Module

This module provides functionality to reconstruct and visualize charts from extracted JSON data.
It supports multiple chart types (bar, pie, scatter, line) and includes an LLM-based fallback
mechanism for handling other unsupported chart types (or cases when chart classifier API fails).

Key Features:
- Load chart metadata from JSON files
- Draw various chart types using matplotlib
- LLM-powered code generation for unsupported chart scenarios
- Automatic fallback to AI-generated visualization code if standard drawing fails

"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional
from langchain_community.llms import Ollama


def get_charts_from_json(json_path: str) -> list:
    """
    Load and parse chart data from the JSON file (built in Chart Extraction Module).
    
    Args:
        json_path (str): Path to the JSON file containing chart metadata.
        
    Returns:
        list: Parsed chart data from the JSON file.
    """
    with open(json_path, "r", encoding='utf-8') as json_file:
        chart = json.load(json_file)

    return chart


def ensure_output_dir(save_path: Optional[str]) -> None:
    """
    Create output directory if it does not exist.
    
    Extracts the directory path from the given file path and creates it
    
    Args:
        save_path (Optional[str]): Full file path including filename. If None, returns without action.
        
    Returns:
        None
    """
    if not save_path:
        return
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

def draw_barchart(chart: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Draw a bar chart and save it to a file.
    
    Expected chart dictionary structure:
    {
        "x_labels": list of category labels,
        "categories": list of data series names,
        "data": dict mapping category names to lists of values,
        "chart_title": str title of the chart
    }
    
    Args:
        chart (Dict[str, Any]): Dictionary containing chart data and metadata.
        save_path (Optional[str]): File path to save the chart image.
    """
    x = np.arange(len(chart["x_labels"]))
    width = 0.2
    categories = chart["categories"]
    fig, ax = plt.subplots()

    for i, cat in enumerate(categories):
        values = chart["data"][cat]
        ax.bar(x + i * width, [float(v) if v is not None else 0 for v in values], width, label=cat)

    ax.set_title(chart["chart_title"])
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(chart["x_labels"])
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

def draw_piechart(chart: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Draw a pie chart and save it to a file.
    
    Expected chart dictionary structure:
    {
        "values": list of numeric values for pie slices,
        "labels": list of labels for each slice,
        "chart_title": str title of the chart
    }
    
    Args:
        chart (Dict[str, Any]): Dictionary containing pie chart data and metadata.
        save_path (Optional[str]): File path to save the chart image.
    """
    plt.pie([float(v) for v in chart["values"]], labels=chart["labels"], autopct="%1.1f%%", startangle=90)
    plt.title(chart["chart_title"])

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

def draw_scatterplot(chart: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Draw a scatter plot and save it to a file.
    
    Expected chart dictionary structure:
    {
        "x_values": list of x-axis numeric values,
        "y_values": list of y-axis numeric values,
        "x_label": str label for x-axis,
        "y_label": str label for y-axis,
        "chart_title": str title of the chart
    }
    
    Args:
        chart (Dict[str, Any]): Dictionary containing scatter plot data and metadata.
        save_path (Optional[str]): File path to save the chart image.

    """
    plt.scatter([float(x) for x in chart["x_values"]], [float(y) for y in chart["y_values"]])
    plt.xlabel(chart["x_label"])
    plt.ylabel(chart["y_label"])
    plt.title(chart["chart_title"])
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

def draw_linechart(chart: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Draw a line chart with multiple series and save it to a file.
    
    Expected chart dictionary structure:
    {
        "x_values": list of x-axis values (labels or numeric),
        "series": dict mapping series names to lists of y-values,
        "x_label": str label for x-axis,
        "chart_title": str title of the chart
    }
    
    Args:
        chart (Dict[str, Any]): Dictionary containing line chart data and metadata.
        save_path (Optional[str]): File path to save the chart image.
    """
    x = chart["x_values"]

    for label, y_values in chart["series"].items():
        plt.plot(x, [float(y) for y in y_values], label=label, marker='o')

    plt.title(chart["chart_title"])
    plt.xlabel(chart["x_label"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)


def draw_chart_from_info(chart_info: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """
    Route chart drawing based on chart type.
    
    Examines the 'chart_type' field in chart_info and calls the appropriate
    specialized drawing function (bar, pie, scatter, or line chart).
    
    Args:
        chart_info (Dict[str, Any]): Dictionary containing chart data with 'chart_type' field.
        save_path (Optional[str]): File path to save the chart image.
    """
    chart_type = chart_info.get('chart_type')
    if chart_type == "Bar Chart":
        draw_barchart(chart_info, save_path)
    elif chart_type == "Pie Chart":
        draw_piechart(chart_info, save_path)
    elif chart_type == "Scatter Plot":
        draw_scatterplot(chart_info, save_path)
    elif chart_type == "Line Chart":
        draw_linechart(chart_info, save_path)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}")


def draw_chart(image_number, save_path: str = "reconstructed_charts/reconstructed_chart.png"):
    """
    Draw a chart by index from the extracted images JSON file.
    
    Loads chart metadata from the extraction results JSON and reconstructs the chart
    at the specified index, with fallback to LLM-generated code if needed.
    
    Args:
        image_number: Index (1-based) of the chart to draw from extracted_images/extraction_results.json.
        save_path (str): Output file path for the reconstructed chart image.
            Defaults to "reconstructed_charts/reconstructed_chart.png".
        
    Returns:
        dict: Result dictionary with keys:
            - "image_path": Path where the chart was saved
            - "generated_code": LLM-generated code if fallback was used, None otherwise
            - "used_fallback": Boolean indicating if LLM fallback was triggered
            - "error": Error message if any occurred, None otherwise
            
    Raises:
        ValueError: If image_number is not an integer or out of valid range.
    """
    ensure_output_dir(save_path)
    try:
        index = int(image_number)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_number must be an integer") from exc

    charts = get_charts_from_json("extracted_images/extraction_results.json")
    if index < 1 or index > len(charts):
        raise ValueError(f"image_number {index} is out of range (1-{len(charts)})")

    chart_info = charts[index - 1]
    return draw_chart_with_fallback(chart_info, save_path=save_path)


def generate_chart_code(
    chart_info: Dict[str, Any],
    model: str = "mistral",
    save_path: Optional[str] = None,
    previous_error: Optional[str] = None,
    generated_code: Optional[str] = None,
) -> str:
    """
    Generate Python visualization code using an LLM model.
    
    Uses Ollama LLM to generate matplotlib code that reconstructs a chart based on
    its metadata. Supports error recovery with retry attempts that include previous
    error messages for correction.
    
    Args:
        chart_info (Dict[str, Any]): Dictionary containing chart metadata and data.
        model (str): LLM model name to use (default: "mistral").
        save_path (Optional[str]): Output file path. If provided, code will save to this path.
        previous_error (Optional[str]): Error message from a previous attempt for LLM correction.
        generated_code (Optional[str]): Previously generated code for reference in error recovery.
        
    Returns:
        str: Generated Python code as a string, cleaned of markdown formatting.
    """
    table_data = chart_info.get("data_table")
    if isinstance(table_data, list):
        table_text = "\n".join(str(row) for row in table_data)
    else:
        table_text = json.dumps(table_data, ensure_ascii=False)

    chart_json = json.dumps(chart_info, ensure_ascii=False)
    chart_type = chart_info.get("chart_type", "unknown")
    chart_title = chart_info.get("chart_title", "")

    guidance = """
You are a Python data visualization assistant. Use matplotlib to rebuild the chart described below.
- Always import matplotlib.pyplot as plt.
- Import numpy as np only when needed.
- Work exclusively with the dictionary named chart_info (already provided in scope).
- Never reference variables named json, data_json, or similar unless you define them.
- Clean numeric strings (%, commas, text) before plotting.
- Assume an OUTPUT_PATH variable may be defined; if it is, always call plt.savefig(OUTPUT_PATH) and then plt.close().
- Return only executable Python code, without Markdown fences.
- Return ONLY the code, no explanations.
"""

    prompt = (
        f"{guidance}\n"
        f"Chart type: {chart_type}\n"
        f"Chart title: {chart_title}\n"
        f"Table data:\n{table_text}\n"
        f"Full chart JSON: {chart_json}\n"
    )

    if save_path:
        prompt += f"OUTPUT_PATH represents '{save_path}'. Always save the chart to this file before closing the figure.\n"

    if previous_error:
        prompt += (
            "The prior attempt failed. Review the error message below and return ONLY the corrected Python code. NO explanations or comments.\n"
            f"Previously generated code{generated_code}"
            f"\nPrevious error: {previous_error}\n"
        )

    prompt += "Write the code now."

    llm = Ollama(model=model)
    generated_code = llm.invoke(prompt)
    cleaned = generated_code.replace("```python", "").replace("```", "").strip()
    return cleaned

def cut_until_first_import_block(text: str) -> str:
    """
    Extract and clean code from LLM-generated output.
    
    Finds the first Python import statement and removes markdown code fences
    to extract clean executable code.

    Handles cases when the LLM output includes explanations or introductions.
    
    Args:
        text (str): Raw text output from LLM that may contain markdown fences.
        
    Returns:
        str: Cleaned code starting from the first 'import' statement.
    """
    start = text.find("import")
    if start != -1:
        return text[start:].replace("```", "").strip()
    return text.replace("```", "").strip()

def draw_chart_with_fallback(
    chart_info: Dict[str, Any],
    save_path: Optional[str] = None,
    model: str = "mistral",
) -> Dict[str, Any]:
    """
    Attempt to draw a chart with automatic fallback to LLM code generation.
    
    First tries to draw the chart using standard matplotlib functions. If that fails,
    triggers an LLM to generate custom visualization code, with to 2 retry attempts
    if the generated code contains errors.
    
    Args:
        chart_info (Dict[str, Any]): Dictionary containing chart data and metadata.
        save_path (Optional[str]): File path to save the chart image. If None, uses default path.
        model (str): LLM model name to use for code generation (default: "mistral").
        
    Returns:
        Dict[str, Any]: Result dictionary with keys:
            - "image_path": Path where the chart was saved
            - "generated_code": Generated code from LLM if fallback was used, None otherwise
            - "used_fallback": Boolean indicating if LLM fallback was triggered
            - "error": Error message if occurred, None if successful
    """
    ensure_output_dir(save_path)
    result = {
        "image_path": save_path,
        "generated_code": None,
        "used_fallback": False,
        "error": None,
    }

    try:
        draw_chart_from_info(chart_info, save_path)
        return result
    except Exception as base_error:
        result["used_fallback"] = True
        print(f"Chart drawing failed: {base_error}")
        print("Calling LLM to generate chart code...")

        target_path = save_path or "reconstructed_charts/reconstructed_chart.png"
        result["image_path"] = target_path
        ensure_output_dir(target_path)

        fallback_error = str(base_error)
        attempt_error = fallback_error

        for attempt in range(2):
            try:
                generated_code = generate_chart_code(
                    chart_info,
                    model=model,
                    save_path=target_path,
                    previous_error=attempt_error if attempt > 0 else fallback_error,
                    generated_code=result["generated_code"] if attempt > 0 else None
                )
                cleaned_code = cut_until_first_import_block(generated_code)
                result["generated_code"] = cleaned_code

                exec_globals: Dict[str, Any] = {
                    "__builtins__": __builtins__,
                    "plt": plt,
                    "np": np,
                }
                exec_locals: Dict[str, Any] = {
                    "chart_info": chart_info,
                    "OUTPUT_PATH": target_path,
                }

                exec(cleaned_code, exec_globals, exec_locals)
                plt.close("all")

                if not os.path.exists(target_path):
                    raise FileNotFoundError(
                        f"Generated code executed but did not create {target_path}"
                    )

                result["error"] = None
                return result
            except Exception as exec_error:
                attempt_error = f"{attempt_error}; fallback failed: {exec_error}"
                result["error"] = attempt_error

        print("LLM fallback failed completely.")

    return result
