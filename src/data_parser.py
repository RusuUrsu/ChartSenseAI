"""
General Chart Data Parser
=========================
A unified solution for parsing chart data from various formats into a consistent structure.
"""

import re
from typing import List, Dict, Any, Optional, Union


def clean_value(value: str) -> str:
    """Clean a value by removing common formatting characters."""
    if not value:
        return ""
    return value.strip().replace('%', '').replace(',', '').replace('|', '')


def is_numeric(value: str) -> bool:
    """Check if a string represents a numeric value."""
    try:
        float(clean_value(value))
        return True
    except (ValueError, TypeError):
        return False


def parse_table_data(data_table: Union[List[str], str]) -> Dict[str, Any]:
    """
    Parse chart data table into a unified format.

    Returns:
    {
        "headers": [list of column headers],
        "x_key": "name of x-axis column (if exists)",
        "data": {
            "label1": [values],
            "label2": [values],
            ...
        }
    }

    For X-Y scatter plots:
    {
        "headers": ["x_label", "y_label"],
        "x_key": None,
        "data": {
            "x_label": [x_values],
            "y_label": [y_values]
        }
    }
    """

    # Convert to list if string
    if isinstance(data_table, str):
        lines = [line.strip() for line in data_table.split('\n') if line.strip()]
    else:
        lines = [line.strip() for line in data_table if line.strip()]

    if len(lines) < 2:
        return {"headers": [], "x_key": None, "data": {}}

    # Find the actual header row (skip TITLE row)
    header_row_idx = 0
    for i, line in enumerate(lines):
        if line.upper().startswith('TITLE'):
            continue
        # Look for a row that contains column separators and potential headers
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) > 1:
            header_row_idx = i
            break

    if header_row_idx >= len(lines):
        return {"headers": [], "x_key": None, "data": {}}

    # Parse headers
    header_line = lines[header_row_idx]
    headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]

    # Remove empty first column if it exists
    if headers and not headers[0]:
        headers = headers[1:]

    if not headers:
        return {"headers": [], "x_key": None, "data": {}}

    # Collect all data rows
    data_rows = []
    for i in range(header_row_idx + 1, len(lines)):
        cells = [cell.strip() for cell in lines[i].split('|')]
        # Remove empty cells and clean
        cells = [cell for cell in cells if cell.strip()]
        if cells:
            data_rows.append(cells)

    if not data_rows:
        return {"headers": headers, "x_key": None, "data": {}}

    # Detect chart type and parse accordingly
    return _parse_data_by_structure(headers, data_rows)


def _parse_data_by_structure(headers: List[str], data_rows: List[List[str]]) -> Dict[str, Any]:
    """Parse data based on the structure detected."""

    # Case 1: Scatter Plot - Two columns with alternating X/Y labels and values
    if _is_scatter_plot_format(headers, data_rows):
        return _parse_scatter_plot(headers, data_rows)

    # Case 2: Single value per category (Pie Chart)
    if len(headers) == 1 or _is_single_category_format(data_rows):
        return _parse_single_category(headers, data_rows)

    # Case 3: Multi-series data (Bar Chart, Line Chart)
    return _parse_multi_series(headers, data_rows)


def _is_scatter_plot_format(headers: List[str], data_rows: List[List[str]]) -> bool:
    """Detect if this is a scatter plot format with alternating labels/values."""
    if len(headers) != 2:
        return False

    # Check if first column contains repeated labels
    first_col_values = [row[0] for row in data_rows if len(row) > 0]
    unique_values = set(first_col_values)

    # If we have alternating labels, it's likely a scatter plot
    return len(unique_values) <= 2 and len(first_col_values) > 2


def _is_single_category_format(data_rows: List[List[str]]) -> bool:
    """Check if this is single category format (like pie chart)."""
    # Single category if each row has exactly 2 elements (category, value)
    return all(len(row) == 2 for row in data_rows if row)


def _parse_scatter_plot(headers: List[str], data_rows: List[List[str]]) -> Dict[str, Any]:
    """Parse scatter plot data with alternating X/Y values."""
    x_label, y_label = headers[0], headers[1]
    x_values = []
    y_values = []

    # Parse alternating pattern
    for row in data_rows:
        if len(row) >= 2:
            label = row[0]
            value = clean_value(row[1])

            if x_label.lower() in label.lower():
                if is_numeric(value):
                    x_values.append(float(value))
            elif y_label.lower() in label.lower():
                if is_numeric(value):
                    y_values.append(float(value))

    return {
        "headers": [x_label, y_label],
        "x_key": None,
        "data": {
            x_label: x_values,
            y_label: y_values
        }
    }


def _parse_single_category(headers: List[str], data_rows: List[List[str]]) -> Dict[str, Any]:
    """Parse single category data (pie chart format)."""
    categories = []
    values = []

    for row in data_rows:
        if len(row) >= 2:
            category = row[0]
            value = clean_value(row[1])

            categories.append(category)
            if is_numeric(value):
                values.append(float(value))
            else:
                values.append(value)

    # Use the first header as the data key, or default to "values"
    data_key = headers[0] if headers else "values"

    return {
        "headers": ["category", data_key],
        "x_key": "category",
        "data": {
            "category": categories,
            data_key: values
        }
    }


def _parse_multi_series(headers: List[str], data_rows: List[List[str]]) -> Dict[str, Any]:
    """Parse multi-series data (bar chart, line chart format)."""
    if not headers:
        return {"headers": [], "x_key": None, "data": {}}

    # First column is typically the x-axis (categories/time)
    x_key = headers[0]
    data_headers = headers[1:]  # Remaining headers are data series

    # Initialize data structure
    data = {x_key: []}
    for header in data_headers:
        data[header] = []

    # Parse each row
    for row in data_rows:
        if not row:
            continue

        # First cell is x-axis value
        x_value = clean_value(row[0])
        data[x_key].append(x_value)

        # Remaining cells are data values
        for i, header in enumerate(data_headers):
            if i + 1 < len(row):
                value = clean_value(row[i + 1])
                if value == '-' or value == '':
                    data[header].append(None)
                elif is_numeric(value):
                    data[header].append(float(value))
                else:
                    data[header].append(value)
            else:
                data[header].append(None)

    return {
        "headers": headers,
        "x_key": x_key,
        "data": data
    }


def extract_chart_data(chart_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to extract chart data from chart info.

    Args:
        chart_info: Dictionary containing 'data_table' and other chart metadata

    Returns:
        Parsed data in unified format
    """
    data_table = chart_info.get('data_table', [])
    parsed_data = parse_table_data(data_table)

    # Add metadata
    parsed_data['chart_type'] = chart_info.get('chart_type', '')
    parsed_data['chart_title'] = chart_info.get('chart_title', '')

    return parsed_data


# Example usage and testing
if __name__ == "__main__":
    # Test with your examples

    # Pie Chart example
    pie_data = [
        "TITLE | % of Students Enrolled",
        "Humanities | 22% ",
        " Business | 30% ",
        " Education | 13% ",
        " Engineering | 20% ",
        " Architecture | 15%"
    ]

    # Bar Chart example
    bar_data = [
        "TITLE | Car Ownership among MHP employees",
        "Year | Mercedes | Bmw | Audi | Porsche",
        "Angajat | 4.30 | 2.44 | 2.00 | 3.02",
        "Manager | 2.50 | 4.40 | 2.00 | 2.82",
        "Director | 3.50 | 1.80 | 3.03 | 2.30",
        "Boss | 4.50 | 2.80 | 5.00 | 5.00",
        "Staff | 2.30 | 2.80 | 4.80 | -"
    ]

    # Line Chart example
    line_data = [
        "TITLE | Wildlife Population ",
        "  | Bears | Dolphins | Whales ",
        " 2020 | 5 | 152 | 79 ",
        " 2021 | 45 | 76 | 52 ",
        " 2022 | 94 | 29 | 99 ",
        " 2023 | 116 | 10 | 73 ",
        " 2024 | 136 | 4 | 92 ",
        " 2025 | 183 | 0 | 69"
    ]

    # Scatter Plot example
    scatter_data = [
        "TITLE |  ",
        " Height (m) | Diameter (cm) ",
        " Diameter (cm) | 2.4 ",
        " Height (m) | 3.25 ",
        " Diameter (cm) | 3.24 ",
        " Height (m) | 3.15"
    ]

    print("Pie Chart:")
    print(parse_table_data(pie_data))
    print("\nBar Chart:")
    print(parse_table_data(bar_data))
    print("\nLine Chart:")
    print(parse_table_data(line_data))
    print("\nScatter Plot:")
    print(parse_table_data(scatter_data))
