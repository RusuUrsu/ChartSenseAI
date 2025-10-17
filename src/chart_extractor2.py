"""
Chart Extraction Module
- extracts chart information from documents:
 builds a JSON file with chart data of all charts found in the document.
 this JSON file is then used to build a vector database for chart question answering
 and to help the local LLM answer chart QAs accurately.
"""

import base64
import json
import logging
import os
import re
from os.path import exists, join, abspath, splitext, basename
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import torch
import fitz
import nyckel
from PIL import Image
from transformers import (
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
)
from gradio_client import Client, handle_file
import comtypes.client
from docx2pdf import convert

logger = logging.getLogger(__name__)
MIN_DRAWING_WIDTH = 50
MIN_DRAWING_HEIGHT = 50
MIN_DRAWING_AREA = 2500
BBOX_PADDING = 3
PIXMAP_SCALE = 2
TEXT_SEARCH_PADDING = 50
PATH_GROUPING_THRESHOLD = 30


def get_chart_type(image_path: str) -> str:
    """
    Identify the type of chart in an image using Nyckel API.

    Args:
        image_path (str): Path to the chart image file.

    Returns:
        str: The identified chart type (e.g., "Bar Chart") or "unknown" if
             identification fails.

    Note:
        Requires valid Nyckel API credentials.
        Nyckel API recognizes multiple chart types. The code includes special implementations
        for common chart types like Bar Chart, Pie Chart, Line Chart, and Scatter Plot.
        (deterministic chart data parsing and redrawing logic is implemented separately.) 
    """
    nyckel_client_id = os.getenv('NYCKEL_CLIENT_ID')
    nyckel_client_secret = os.getenv('NYCKEL_CLIENT_SECRET')

    try:
        credentials = nyckel.Credentials(
            client_id=nyckel_client_id,
            client_secret=nyckel_client_secret,
        )
        with open(image_path, 'rb') as f:
            image_data = f.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{image_base64}"
        result = nyckel.invoke('chart-types-identifier', data_uri, credentials)
        return result.get('labelName', 'unknown')
    except Exception as e:
        logger.warning(f"Failed to identify chart type: {e}")
        return "unknown"




def get_chart_structure(image_path: str) -> str:
    """
    Extract the underlying data structure from a chart image.

    Uses the Pix2Struct model (google/deplot) to generate a formatted
    table representation of the chart's data. The model performs optical chart
    recognition to convert visual charts into structured tabular data.

    Args:
        image_path (str): Path to the chart image file. Supports PNG, JPG, etc.

    Returns:
        str: A formatted table representation of the chart data, or empty string
             if extraction fails.
    """
    try:
        torch.set_default_device("cpu")
        processor = Pix2StructProcessor.from_pretrained('google/deplot')
        model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

        image = Image.open(image_path).convert('RGB')
        inputs = processor(
            images=image,
            text="Generate underlying data table of the figure below:",
            return_tensors="pt"
        )
        predictions = model.generate(**inputs, max_new_tokens=512)
        result = processor.decode(predictions[0], skip_special_tokens=True)
        return result.replace('<0x0A>', "\n")
    except Exception as e:
        logger.error(f"Error extracting chart structure: {e}")
        return ""


def parse_info_for_bar_chart(chart_info: list) -> dict:
    """
    Parse bar chart data from structured table format.

    Expects a list of pipe-delimited rows where the first column represents
    x-axis labels and subsequent columns represent data series.

    Args:
        chart_info (list): List of pipe-delimited strings representing chart data.
                          First row is header, subsequent rows are data points.

    Returns:
        dict: Dictionary with keys:
            - x_labels (list): Labels for x-axis categories
            - categories (list): Names of data series
            - data (dict): Mapping of category names to lists of values
    """
    header = chart_info[0].split('|')
    x_key = header[0].strip()
    categories = [h.strip() for h in header[1:]]

    x_labels = []
    data = {cat: [] for cat in categories}

    for row in chart_info[1:]:
        parts = row.split('|')
        x_label = parts[0].strip()
        values = extract_values([v.strip() for v in parts[1:]])

        x_labels.append(x_label)
        for cat, val in zip(categories, values):
            data[cat].append(None if val in (0, "-") else val)

    return {
        "x_labels": x_labels,
        "categories": categories,
        "data": data,
    }

def extract_values(values: list) -> list:
    """
    Clean and convert string values to floats.

    Removes non-numeric characters (except decimal points and commas) and
    converts the resulting strings to floating-point numbers.

    Args:
        values (list): List of string values potentially containing non-numeric
                      characters (%, commas, special characters, etc.).

    Returns:
        list: List of float values. Non-numeric strings become 0.0.
    """
    cleaned = [re.sub(r'[^0-9,\.]', '', val) for val in values]
    return [float(val) if val else 0.0 for val in cleaned]

def parse_info_for_pie_chart(chart_info: list) -> dict:
    """
    Parse pie chart data from structured format.

    Expects a list of pipe-delimited strings where each row contains a label
    and corresponding percentage or value.

    Args:
        chart_info (list): List of pipe-delimited strings. Each row contains
                          two pipe-separated values: label and value.

    Returns:
        dict: Dictionary with keys:
            - labels (list): Category labels for pie slices
            - values (list): Numeric values for each slice
    """
    labels = [c.split('|')[0].strip() for c in chart_info]
    values = [c.split('|')[1].strip().replace('%', "") for c in chart_info]
    return {
        'labels': labels,
        'values': extract_values(values),
    }

def parse_info_for_scatter_chart(chart_info: list) -> dict:
    """
    Parse scatter plot data from structured format.

    Args:
        chart_info (list): List of pipe-delimited strings where first row
                          contains axis labels, and subsequent rows contain
                          x,y coordinate pairs.

    Returns:
        dict: Dictionary with keys:
            - x_label (str): Label for x-axis
            - y_label (str): Label for y-axis
            - x_values (list): List of x-coordinates
            - y_values (list): List of y-coordinates
    """
    labels = chart_info[0].split('|')
    series = [chart_info[i].split('|') for i in range(1, len(chart_info))]
    
    return {
        'x_label': labels[0].strip(),
        'y_label': labels[1].strip(),
        'x_values': extract_values([s[0].strip() for s in series]),
        'y_values': extract_values([s[1].strip() for s in series]),
    }

def parse_info_for_line_chart(chart_info: list) -> dict:
    """
    Parse line chart data from structured format.

    Supports multiple data series. First row contains headers with first column
    as x-axis label and remaining columns as series names.

    Args:
        chart_info (list): List of pipe-delimited strings representing time series
                          or x-y data with multiple series.

    Returns:
        dict: Dictionary with keys:
            - x_label (str): Label for x-axis (e.g., "Year")
            - categories (list): Names of data series
            - x_values (list): Values for x-axis
            - series (dict): Mapping of series names to lists of values
    """
    categories = chart_info[0].split('|')
    x_label = categories[0].strip()
    categories = [cat.strip() for cat in categories[1:]]
    series = [chart_info[i].split('|') for i in range(1, len(chart_info))]
    
    return {
        'x_label': x_label,
        'categories': categories,
        'x_values': [x[0].strip() for x in series],
        'series': {
            categories[i]: extract_values([s[i+1].strip() for s in series])
            for i in range(len(categories))
        },
    }


CHART_PARSERS = {
    'Pie Chart': parse_info_for_pie_chart,
    'Bar Chart': parse_info_for_bar_chart,
    'Scatter Plot': parse_info_for_scatter_chart,
    'Line Chart': parse_info_for_line_chart,
}


def get_chart_info(image_path: str, nearby_text: str = "") -> dict:
    """
    Extract and parse comprehensive chart information from an image.

    This is the main entry point for chart analysis. It orchestrates the
    following steps:
    1. Extract chart data table using Pix2Struct model
    2. Identify chart type using Nyckel API
    3. Extract title
    4. Parse data according to chart type
    5. Compile results with metadata

    Args:
        image_path (str): Path to the chart image file.
        nearby_text (str, optional): Text appearing near the chart in the document.
                                    Defaults to empty string.

    Returns:
        dict: Comprehensive chart information containing:
            - data_table (list): Raw table rows
            - file_path (str): Original image path
            - chart_type (str): Identified chart type
            - nearby_text (str): Context text
            - chart_title (str): Extracted title or "unavailable"
            - Type-specific fields (x_labels, categories, data, etc.)
    """
    chart_structure = get_chart_structure(image_path)
    chart_lines = chart_structure.split('\n')

    chart_info = {
        'data_table': chart_lines,
        'file_path': image_path,
        'chart_type': get_chart_type(image_path),
        'nearby_text': nearby_text,
    }

    try:
        chart_info['chart_title'] = chart_lines[0].split('|')[1].strip()
    except (IndexError, AttributeError):
        chart_info['chart_title'] = "unavailable"

    try:
        chart_type = chart_info['chart_type']
        if chart_type in CHART_PARSERS and len(chart_lines) > 1:
            parser = CHART_PARSERS[chart_type]
            chart_info.update(parser(chart_lines[1:]))
    except Exception as e:
        logger.error(f"Error parsing chart info: {e}")

    return chart_info


def extract_vectorial_drawings(page, output_dir: str, page_number: int, global_counter: int, existing_image_bboxes: list = None) -> list:
    """
    Extract all vectorial drawings from a PDF page, excluding those that overlap with embedded images.
    Vectorial drawings: Charts created when (example) Word converts to PDF and chart was inserted into 
    Word from Excel.
    Embedded images: Charts inserted as images in the document (word/pdf).

    The algorithm groups nearby paths together to avoid fragmenting a single chart
    into multiple parts (e.g., treating individual bar segments or axis lines as
    separate images). Each identified chart is treated as one unified drawing.

    Duplicate Detection:
        When charts are inserted from Excel into Word then converted to PDF, they
        appear as both embedded images AND vectorial drawings. This function skips
        any vectorial drawing that overlaps with an already-extracted image to
        prevent duplicate chart extraction.
        
        Additionally, multiple vectorial representations of the same chart are
        deduplicated by comparing bounding boxes and keeping only the first one.

    Args:
        page: PyMuPDF page object.
        output_dir (str): Directory to save extracted drawing images.
        page_number (int): Index of the page being processed (0-based).
        global_counter (int): Global image counter for naming consistency.
        existing_image_bboxes (list): List of bounding boxes of already-extracted images.
                                     Defaults to empty list if None.

    Returns:
        list: List of chart info dicts for all extracted vectorial drawings.
              Multiple drawings on the same page are all extracted (not just largest).
    """
    if existing_image_bboxes is None:
        existing_image_bboxes = []

    drawings = []
    paths = page.get_drawings()

    if not paths:
        return drawings

    drawing_groups = group_nearby_paths(paths)
    valid_groups = []

    for group in drawing_groups:
        bbox = calculate_group_bbox(group)
        if not bbox:
            continue

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height

        if width > MIN_DRAWING_WIDTH and height > MIN_DRAWING_HEIGHT and area > MIN_DRAWING_AREA:
            valid_groups.append((group, bbox, area))

    if not valid_groups:
        return drawings

    # Deduplicate drawings on the same page by bbox similarity
    deduplicated_groups = _deduplicate_drawing_bboxes(valid_groups)

    for drawing_index, (_, bbox, _) in enumerate(deduplicated_groups, 1):
        if _drawing_overlaps_with_image(bbox, existing_image_bboxes):
            logger.debug(f"Skipping vectorial drawing on page {page_number + 1} (overlaps with embedded image)")
            continue

        expanded_bbox = fitz.Rect(
            max(0, bbox[0] - BBOX_PADDING),
            max(0, bbox[1] - BBOX_PADDING),
            min(page.rect.width, bbox[2] + BBOX_PADDING),
            min(page.rect.height, bbox[3] + BBOX_PADDING),
        )

        mat = fitz.Matrix(PIXMAP_SCALE, PIXMAP_SCALE)
        pix = page.get_pixmap(matrix=mat, clip=expanded_bbox)

        drawing_filename = f"page_{page_number + 1}_drawing_{drawing_index}.png"
        drawing_path = join(output_dir, drawing_filename)
        pix.save(drawing_path)
        pix = None

        nearby_text = extract_text_outside_area(page, bbox)
        chart_info = get_chart_info(drawing_path, nearby_text)
        chart_info.update({
            "page_number": page_number + 1,
            "image_number": global_counter + drawing_index - 1,
        })
        drawings.append(chart_info)

    return drawings


def _deduplicate_drawing_bboxes(valid_groups: list, bbox_similarity_threshold: float = 0.95) -> list:
    """
    Deduplicate vectorial drawings that are likely the same visual element.

    When PDF rendering creates multiple paths for the same visual object
    (e.g., different rendering passes or layers), this function detects and
    removes near-duplicate bounding boxes.

    Args:
        valid_groups (list): List of (group, bbox, area) tuples.
        bbox_similarity_threshold (float): IoU threshold for considering bboxes
                                         as duplicates. Defaults to 0.95 (95%
                                         overlap indicates same object).

    Returns:
        list: Deduplicated list of (group, bbox, area) tuples.
    """
    if len(valid_groups) <= 1:
        return valid_groups

    deduplicated = []
    used_indices = set()

    for i, (group_i, bbox_i, area_i) in enumerate(valid_groups):
        if i in used_indices:
            continue

        # Check if this bbox is a duplicate of any already-kept bbox
        is_duplicate = False
        for group_k, bbox_k, area_k in deduplicated:
            iou = _calculate_iou(bbox_i, bbox_k)
            if iou >= bbox_similarity_threshold:
                is_duplicate = True
                logger.debug(f"Skipping duplicate vectorial drawing (IoU: {iou:.2f})")
                break

        if not is_duplicate:
            deduplicated.append((group_i, bbox_i, area_i))
        else:
            used_indices.add(i)

    return deduplicated


def _calculate_iou(bbox1: tuple, bbox2: tuple) -> float:
    """
    Calculate Intersection-over-Union (IoU) between two bounding boxes.

    Args:
        bbox1 (tuple): First bbox as (x0, y0, x1, y1).
        bbox2 (tuple): Second bbox as (x0, y0, x1, y1).

    Returns:
        float: IoU ratio between 0 and 1. Returns 0 if boxes don't intersect.
    """
    x0_1, y0_1, x1_1, y1_1 = bbox1
    x0_2, y0_2, x1_2, y1_2 = bbox2

    inter_x0 = max(x0_1, x0_2)
    inter_y0 = max(y0_1, y0_2)
    inter_x1 = min(x1_1, x1_2)
    inter_y1 = min(y1_1, y1_2)

    if inter_x0 < inter_x1 and inter_y0 < inter_y1:
        intersection_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    else:
        return 0.0

    area1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    area2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def _drawing_overlaps_with_image(drawing_bbox: tuple, image_bboxes: list, overlap_threshold: float = 0.7) -> bool:
    """
    Check if a vectorial drawing overlaps with any embedded image.

    Uses intersection-over-union (IoU) ratio to detect significant overlaps,
    which indicates the drawing is likely the same object as the embedded image.

    Args:
        drawing_bbox (tuple): Drawing bounding box as (x0, y0, x1, y1).
        image_bboxes (list): List of image bounding boxes.
        overlap_threshold (float): Minimum IoU ratio to consider as overlap.
                                 Defaults to 0.7 (70% overlap required).

    Returns:
        bool: True if drawing significantly overlaps with any image.
    """
    if not image_bboxes:
        return False

    dx0, dy0, dx1, dy1 = drawing_bbox

    for img_bbox in image_bboxes:
        ix0, iy0, ix1, iy1 = img_bbox

        inter_x0 = max(dx0, ix0)
        inter_y0 = max(dy0, iy0)
        inter_x1 = min(dx1, ix1)
        inter_y1 = min(dy1, iy1)

        if inter_x0 < inter_x1 and inter_y0 < inter_y1:
            intersection_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
            drawing_area = (dx1 - dx0) * (dy1 - dy0)
            image_area = (ix1 - ix0) * (iy1 - iy0)

            union_area = drawing_area + image_area - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0

            if iou >= overlap_threshold:
                return True

    return False


def extract_text_outside_area(page, drawing_bbox: tuple, padding: int = 50) -> str:
    """
    Extract text near a drawing but not overlapping with it.
    Gives more context about the chart from surrounding text.

    Args:
        page: PyMuPDF page object.
        drawing_bbox (tuple): Bounding box of drawing as (x0, y0, x1, y1).
        padding (int): Pixels to expand search area beyond drawing bbox.
                      Defaults to 50.

    Returns:
        str: Concatenated text found in search area but outside drawing bbox.
    """
    search_area = fitz.Rect(
        max(0, drawing_bbox[0] - padding),
        max(0, drawing_bbox[1] - padding),
        min(page.rect.width, drawing_bbox[2] + padding),
        min(page.rect.height, drawing_bbox[3] + padding),
    )

    text_blocks = page.get_text("dict", clip=search_area)
    nearby_text_parts = []
    drawing_rect = fitz.Rect(*drawing_bbox)

    for block in text_blocks.get("blocks", []):
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                span_rect = fitz.Rect(span["bbox"])
                if not span_rect.intersects(drawing_rect):
                    text = span["text"].strip()
                    if text:
                        nearby_text_parts.append(text)

    return " ".join(nearby_text_parts).strip()


def group_nearby_paths(paths: list, distance_threshold: float = 30) -> list:
    """
    Group paths that are close together to form complete drawings.
    Avoids fragmenting single charts into multiple parts.

    Args:
        paths (list): List of PyMuPDF path objects.
        distance_threshold (float): Maximum distance between path centers for
                                   grouping. Defaults to 30 points.

    Returns:
        list: List of path groups, where each group is a list of paths.
    """
    if not paths:
        return []

    groups = []
    used_paths = set()

    for i, path in enumerate(paths):
        if i in used_paths:
            continue

        current_group = [path]
        used_paths.add(i)
        path_bbox = calculate_path_bbox(path)
        
        if not path_bbox:
            continue

        for j, other_path in enumerate(paths):
            if j in used_paths:
                continue

            other_bbox = calculate_path_bbox(other_path)
            if not other_bbox or not paths_are_nearby(path_bbox, other_bbox, distance_threshold):
                continue

            current_group.append(other_path)
            used_paths.add(j)

        groups.append(current_group)

    return groups


def calculate_path_bbox(path: dict) -> tuple | None:
    """
    Calculate bounding box for a single PDF path.

    Handles both simple rect paths and complex paths with multiple items.

    Args:
        path (dict): PyMuPDF path dictionary containing 'rect' or 'items'.

    Returns:
        tuple: Bounding box as (x0, y0, x1, y1), or None if bbox cannot be calculated.
    """
    try:
        if 'rect' in path:
            rect = path['rect']
            return (rect.x0, rect.y0, rect.x1, rect.y1)
        if 'items' in path:
            all_points = []
            for item in path['items']:
                if item[0] in ['l', 'm', 'c']:
                    all_points.extend(item[1:])

            if all_points:
                x_coords = [p.x if hasattr(p, 'x') else p[0] for p in all_points]
                y_coords = [p.y if hasattr(p, 'y') else p[1] for p in all_points]
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    except Exception as e:
        logger.debug(f"Error calculating path bbox: {e}")
    return None


def calculate_group_bbox(path_group: list) -> tuple | None:
    """
    Calculate combined bounding box for a group of paths.

    Finds the minimum bounding rectangle that contains all paths in the group.

    Args:
        path_group (list): List of path objects or dictionaries.

    Returns:
        tuple: Combined bounding box as (x0, y0, x1, y1), or None if group is empty.
    """
    bboxes = [bbox for bbox in (calculate_path_bbox(path) for path in path_group) if bbox]

    if not bboxes:
        return None

    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def paths_are_nearby(bbox1: tuple, bbox2: tuple, threshold: float) -> bool:
    """
    Check if two bounding boxes are within threshold distance.

    Calculates Euclidean distance between bbox centers.

    Args:
        bbox1 (tuple): First bounding box as (x0, y0, x1, y1).
        bbox2 (tuple): Second bounding box as (x0, y0, x1, y1).
        threshold (float): Maximum distance for proximity.

    Returns:
        bool: True if distance > 0 and distance <= threshold.
    """
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2

    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    return 0 < distance <= threshold


def save_extraction_results(results: list, output_dir: str) -> None:
    """
    Save extraction results to JSON file.

    Args:
        results (list): List of chart information dictionaries.
        output_dir (str): Directory to save the results file.

    Saves to: <output_dir>/extraction_results.json
    """
    json_path = join(output_dir, "extraction_results.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2, ensure_ascii=False)


def get_image_regions(pdf_path: str, output_dir: str = "extracted_images") -> list:
    """
    Extract all embedded images and vectorial drawings from a PDF.

    This is the main extraction function that processes an entire PDF and
    returns a list of all discovered charts.

    Args:
        pdf_path (str): Path to the PDF file to process.
        output_dir (str): Directory to save extracted images and results JSON.
                         Defaults to "extracted_images".

    Returns:
        list: List of chart information dictionaries, one per extracted chart.
              Each dict contains chart data, type, title, and metadata.

    Saves:
        - Individual chart images to <output_dir>/page_X_image_Y.{ext}
        - Chart extraction results to <output_dir>/extraction_results.json
    """
    Path(output_dir).mkdir(exist_ok=True)
    logger.info("Processing PDF")

    pdf_doc = fitz.open(pdf_path)
    results = []
    global_image_counter = 0
    image_bboxes = []

    try:
        for page_number in range(len(pdf_doc)):
            page = pdf_doc[page_number]

            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    global_image_counter += 1
                    img_path = extract_image_from_pdf(
                        pdf_doc, xref, page_number, output_dir, global_image_counter
                    )

                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        bbox = (img_rects[0].x0, img_rects[0].y0, img_rects[0].x1, img_rects[0].y1)
                        image_bboxes.append(bbox)
                        nearby_text = extract_text_outside_area(page, bbox)
                        chart_info = get_chart_info(img_path, nearby_text)
                        chart_info.update({
                            "page_number": page_number + 1,
                            "image_number": global_image_counter,
                        })
                        results.append(chart_info)
                except Exception as e:
                    logger.error(f"Error extracting image {img_index + 1} from page {page_number + 1}: {e}")

            try:
                drawings = extract_vectorial_drawings(page, output_dir, page_number, global_image_counter, image_bboxes)
                results.extend(drawings)
                if drawings:
                    global_image_counter += len(drawings)
            except Exception as e:
                logger.error(f"Error extracting drawings from page {page_number + 1}: {e}")
    finally:
        pdf_doc.close()

    save_extraction_results(results, output_dir)
    return results


def extract_image_from_pdf(pdf_doc, xref: int, page_number: int, output_dir: str, global_counter: int) -> str:
    """
    Extract a single image from a PDF by reference.

    Args:
        pdf_doc: PyMuPDF PDF document object.
        xref (int): Cross-reference number of the image in the PDF.
        page_number (int): Page index (0-based).
        output_dir (str): Directory to save extracted image.
        global_counter (int): Global image counter for consistent naming.

    Returns:
        str: Path to the saved image file.
    """
    base_image = pdf_doc.extract_image(xref)
    image_bytes = base_image["image"]
    image_ext = base_image["ext"]

    img_filename = f"page_{page_number + 1}_image_{global_counter}.{image_ext}"
    img_path = join(output_dir, img_filename)

    with open(img_path, "wb") as img_file:
        img_file.write(image_bytes)

    return img_path

def ensure_pdf_extension(output_path: str) -> str:
    """
    Ensure output path has .pdf extension.

    Args:
        output_path (str): File path that may or may not have .pdf extension.

    Returns:
        str: Path with .pdf extension appended if needed.
    """
    if output_path.lower().endswith('.pdf'):
        return output_path
    return splitext(output_path)[0] + '.pdf'


def convert_word_to_pdf(file_path: str, output_path: str) -> str | None:
    """
    Convert a .docx file to PDF using docx2pdf library.

    Args:
        file_path (str): Path to .docx file to convert.
        output_path (str): Output path for PDF file (extension auto-added if needed).

    Returns:
        str | None: Path to created PDF file, or None if conversion fails.
    """
    try:
        output_path = ensure_pdf_extension(output_path)
        convert(file_path, output_path)

        if exists(output_path):
            return output_path
        logger.error(f"PDF conversion completed but file not found at: {output_path}")
        return None
    except Exception as e:
        logger.error(f"Error converting .docx to PDF: {e}")
        return None


def convert_to_pdf_if_needed(file_path: str, temp_dir: str = "temp_pdf") -> str:
    """
    Convert document files to PDF if needed.

    Automatically converts .docx files to PDF format. Returns
    PDF path unchanged if file is already in PDF format.

    Args:
        file_path (str): Path to document file (.pdf, .docx, or .pptx).
        temp_dir (str): Temporary directory for converted files.
                       Defaults to "temp_pdf".

    Returns:
        str: Path to PDF file (original or converted).

    Raises:
        ValueError: If file format is unsupported or conversion fails.
    """
    Path(temp_dir).mkdir(exist_ok=True)
    file_ext = splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        return file_path

    if file_ext not in ['.docx', '.pptx']:
        raise ValueError(f"Unsupported file extension: {file_ext}")

    if file_ext == '.docx':
        pdf_path = convert_word_to_pdf(file_path, file_path)
    else:
        raise ValueError(f"PPTX conversion not yet implemented")

    if not (pdf_path and exists(pdf_path)):
        raise ValueError(f"Conversion failed: {file_ext} → PDF")

    return pdf_path


def parse_file(file_path: str) -> list:
    """
    Parse file and extract all chart information.

    Main entry point for file processing. Handles format conversion,
    PDF extraction, and cleanup.

    Args:
        file_path (str): Path to document file (.pdf, .docx).

    Returns:
        list: List of chart information dictionaries. Empty list if file
              not found or processing fails.

    Processing Steps:
        1. Validate file exists
        2. Convert to PDF if needed
        3. Extract charts from PDF
        4. Clean up temporary files
        5. Return results
    """
    if not exists(file_path):
        logger.error(f"File not found: {file_path}")
        return []

    try:
        pdf_path = convert_to_pdf_if_needed(file_path)
        results = get_image_regions(pdf_path)

        if file_path != pdf_path:
            try:
                os.remove(pdf_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary PDF: {e}")

        return results
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []
