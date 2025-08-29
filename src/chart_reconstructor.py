import matplotlib.pyplot as plt
import numpy as np
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from langchain_community.llms import Ollama
from numpy.f2py.auxfuncs import throw_error

from src.llm import invoke_llm_with_template
from src.prompts import reconstruction_template, parsing_template


def get_charts_from_json(json_path):
    with open(json_path, "r", encoding='utf-8') as json_file:
        chart = json.load(json_file)

    return chart

def draw_barchart(chart, save_path=None):
    x = np.arange(len(chart["x_labels"]))  # the label locations
    width = 0.2  # width of the bars
    categories = chart["categories"]
    fig, ax = plt.subplots()

    for i, cat in enumerate(categories):
        values = chart["data"][cat]
        ax.bar(x + i * width, [float(v) if v is not None else 0 for v in values], width, label=cat)

    # Add labels and title
    ax.set_title(chart["chart_title"])
    ax.set_xticks(x + width * (len(categories) - 1) / 2)
    ax.set_xticklabels(chart["x_labels"])
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

def draw_piechart(chart, save_path=None):
    plt.pie([float(v) for v in chart["values"]], labels=chart["labels"], autopct="%1.1f%%", startangle=90)
    plt.title(chart["chart_title"])
    #plt.axis("equal")

    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

def draw_scatterplot(chart, save_path=None):
    plt.scatter([float(x) for x in chart["x_values"]], [float(y) for y in chart["y_values"]])
    plt.xlabel(chart["x_label"])
    plt.ylabel(chart["y_label"])
    plt.title(chart["chart_title"])
    plt.grid(True)

    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)

def draw_linechart(chart, save_path=None):
    x = chart["x_values"]
    # all_values = []
    # for y_list in chart["series"].values():
    #     all_values.extend(y_list)
    # plt.ylim(0, max(all_values))

    for label, y_values in chart["series"].items():
        plt.plot(x, [float(y) for y in y_values], label=label, marker='o')

    plt.title(chart["chart_title"])
    plt.xlabel(chart["x_label"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show(block=False)


def draw_chart(image_number, save_path="reconstructed_charts/reconstructed_chart.png"):
    chart_info = get_charts_from_json("extracted_images/extraction_results.json")[image_number-1]
    chart_type = chart_info['chart_type']
    if chart_type == "Bar Chart":
        draw_barchart(chart_info, save_path)
    elif chart_type == "Pie Chart":
        draw_piechart(chart_info, save_path)
    elif chart_type == "Scatter Plot":
        draw_scatterplot(chart_info, save_path)
    elif chart_type == "Line Chart":
        draw_linechart(chart_info, save_path)
    else:
        # raise Exception(f"Unknown chart type: {chart_type}")
        print("Unknown chart type")


def draw_chart_with_fallback(chart_info):
    try:
        draw_chart(chart_info)
    except Exception as e:
        print(f"Chart drawing failed: {e}")
        print("Calling LLM to generate chart code...")

        # Prepare data for LLM
        table_data = json.dumps(chart_info.get("data_table", {}))
        chart_type = chart_info.get("chart_type", "unknown")

        # Call LLM with prompt template
        chart_info = {
            "chart_type": chart_type,
            "data_table": table_data
        }
        llm = Ollama(model="mistral")
        prompt = f"""
        Given the following chart type and data table
        chart_type: {chart_type}
        data_table: {table_data}
        Write python matplotlib code to generate the chart.
        Make sure to handle missing values and label the axes and title appropriately.
        Make sure to handle % signs or strings in values.
        Only return the code block without any explanation.
        Your Answer:
        """

        generated_code = llm.invoke(prompt)
        generated_code = generated_code.replace("python", '').replace("`", '').strip()
        print("Generated code:")
        print(generated_code)

        #Optionally execute the generated code
        try:

            exec(generated_code.replace("```", ''))
        except Exception as exec_error:
            print(f"Generated code execution failed: {exec_error}")


def save_charts_to_folder(charts, folder_path="reconstructed_charts"):
    """
    Save all reconstructed charts to the specified folder

    Args:
        charts: List of chart dictionaries
        folder_path: Path to the folder where charts will be saved

    Returns:
        List of paths to saved chart files
    """
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, chart in enumerate(charts):
        try:
            # Generate filename based on chart title or index
            if "chart_title" in chart and chart["chart_title"]:
                # Clean title for filename (replace spaces and special chars)
                title = chart["chart_title"].replace(" ", "_").replace("/", "_").replace("\\", "_")
                title = ''.join(c for c in title if c.isalnum() or c == '_')
                filename = f"{title}_{timestamp}.png"
            else:
                filename = f"chart_{i+1}_{timestamp}.png"

            # Full path to save
            save_path = os.path.join(folder_path, filename)

            # Draw and save chart
            draw_chart(chart, save_path)
            saved_paths.append(save_path)
            print(f"Saved chart to {save_path}")

        except Exception as e:
            print(f"Failed to save chart {i+1}: {e}")

    print(f"Saved {len(saved_paths)} charts to {os.path.abspath(folder_path)}")
    return saved_paths


def main():
    chart_entries = get_charts_from_json("extracted_images/extraction_results.json")

    # Add option to save charts
    save_option = input("Do you want to save the charts to the 'reconstructed_charts' folder? (y/n): ").lower()

    if save_option == 'y':
        # Save all charts
        save_charts_to_folder(chart_entries)
    else:
        # Just display charts
        for chart in chart_entries:
            try:
                draw_chart_with_fallback(chart)
                print(f"Success! for chart:{chart['chart_type']}")
            except Exception as e:
                print(f"Failed to draw chart {chart.get('chart_title', 'unknown')}: {str(e)}")

if __name__ == "__main__":
    main()
