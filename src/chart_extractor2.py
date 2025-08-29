import fitz
import json

import torch
from PIL import Image
import nyckel
import base64
from transformers import Pix2StructForConditionalGeneration,Pix2StructProcessor
import re
from gradio_client import Client, handle_file
from src.data_parser import parse_table_data
import comtypes.client
from docx2pdf import convert

def get_chart_description(image_path):
    client = Client("ahmed-masry/ChartGemma")
    result = client.predict(
        image = handle_file(image_path),
        input_text="Generate description of the figure below",
        api_name="/predict"
    )
    return result

def get_chart_type(image_path):
    try:
        credentials = nyckel.Credentials(client_id='1v0pwr8idgcbbwgm9bgbs2c4rjnc1lsu',
                                         client_secret='tu0i6wx3r88tw5t4iu6s0o8yjvxd4vhsnhhgwxfcddkjtj8mpiu6fzu4m1zazkyv')
        with open(image_path, 'rb') as f:
            image_data = f.read()

        image_base64 = base64.b64encode(image_data).decode('utf-8')
        data_uri = f"data:image/png;base64,{image_base64}"
        res = nyckel.invoke('chart-types-identifier', data_uri, credentials)
        return res['labelName']
    except:
        return "unknown"
    return "Unknown"




def get_chart_structure(image_path):

    try:
        torch.set_default_device("cpu")
        processor = Pix2StructProcessor.from_pretrained('google/deplot')
        model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

        img_path = image_path
        image = Image.open(img_path)
        image = image.convert('RGB')

        inputs = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt")
        predictions = model.generate(**inputs, max_new_tokens=512)
        result = processor.decode(predictions[0], skip_special_tokens=True)

        result = result.replace('<0x0A>', "\n")
    except Exception as e:
        print(f"Error in get_chart_structure: {e}")
        result = ""

    return result

def parse_info_general(chart_info):
    """
    Fixed general parser that uses the new unified data parser.
    This replaces the buggy original implementation.
    """
    # Use the new unified parser
    parsed_data = parse_table_data(chart_info)

    # Convert to the expected format for backward compatibility
    result = {}
    result['header'] = parsed_data.get('headers', [])

    # Convert the unified data format back to the old format for compatibility
    data_dict = {}
    chart_data = parsed_data.get('data', {})
    x_key = parsed_data.get('x_key')

    if x_key and x_key in chart_data:
        # Multi-series format (bar chart, line chart style)
        x_values = chart_data[x_key]
        for series_name, series_values in chart_data.items():
            if series_name != x_key:
                for i, (x_val, y_val) in enumerate(zip(x_values, series_values)):
                    if y_val is not None:
                        data_dict[f"{x_val}, {series_name}"] = str(y_val)
    else:
        # Single category format or X-Y format
        for key, values in chart_data.items():
            if isinstance(values, list):
                for i, value in enumerate(values):
                    if value is not None:
                        data_dict[f"{key}_{i}"] = str(value)
            else:
                data_dict[key] = str(values)

    result['data'] = data_dict
    return result

# Fixed version of the original function (for reference, but recommend using above)
def parse_info_general_fixed(chart_info):
    """
    Fixed version of the original buggy function.
    """
    def is_header_row(row):
        return any(all(not c.isalpha() and c not in "     " for c in cell) for cell in row)

    if not chart_info or len(chart_info) == 0:
        return {'header': [], 'data': {}}

    header = chart_info[0].split("|")
    header = [h.strip() for h in header if h.strip()]  # Clean headers

    if is_header_row(header):
        start = 0
    else:
        start = 1
        if len(chart_info) > 1:
            # Fixed: chart_info is a list, not a string with .split method
            col1 = [chart_info[i].split('|')[0].strip() for i in range(1, len(chart_info))]

            if is_header_row(col1) or (header and header[0] in "   "):
                if header and not header[0].strip():  # Remove empty first header
                    header = header[1:]

                rows = {}
                for i in range(len(col1)):
                    row_parts = chart_info[i + 1].split('|')
                    for j in range(min(len(header), len(row_parts) - 1)):
                        key = f'{col1[i]}, {header[j]}'
                        value = row_parts[j + 1].strip()
                        rows[key] = value
            else:
                rows = {}
                for i in range(1, len(chart_info)):
                    row_parts = chart_info[i].split('|')
                    for j in range(min(len(header), len(row_parts))):
                        key = header[j]
                        value = row_parts[j].strip()
                        if key not in rows:
                            rows[key] = []
                        rows[key].append(value)

    result = {}
    result['header'] = header
    result['data'] = rows if 'rows' in locals() else {}

    return result

def parse_info_for_bar_chart(chart_info):
    result = {}

    # Step 2: Get categories (car brands)
    header = chart_info[0].split('|')
    x_key = header[0].strip()  # Usually something like "Year" or "Role"
    categories = [h.strip() for h in header[1:]]

    # Step 3: Parse data rows
    x_labels = []
    data = {cat: [] for cat in categories}

    for row in chart_info[1:]:
        parts = row.split('|')
        x_label = parts[0].strip()
        values = extract_values([v.strip() for v in parts[1:]])

        x_labels.append(x_label)
        for cat, val in zip(categories, values):
            # Convert to float or None
            if val == "-" or val == "":
                data[cat].append(None)
            else:
                data[cat].append(val)

    # Step 4: Package into JSON structure
    result = {
        "x_labels": x_labels,
        "categories": categories,
        "data": data
    }

    return result

def extract_values(values):
    values = [re.sub(r'[^0-9,\.]', '', val) for val in values]
    values = [float(val) if val != '' else 0 for val in values]
    return values

def parse_info_for_pie_chart(chart_info):
    result = {}
    result['labels'] = [c.split('|')[0].strip() for c in chart_info]

    values = [c.split('|')[1].strip().replace('%', "") for c in chart_info]
    # values = [re.sub(r'[^0-9,\.]', '', val) for val in values]
    result['values'] = extract_values(values)

    return result

def parse_info_for_scatter_chart(chart_info):
    result = {}
    labels = chart_info[0].split('|')
    result['x_label'] = labels[0].strip()
    result['y_label'] = labels[1].strip()

    series = [chart_info[i].split('|') for i in range(1,len(chart_info))]
    result['x_values'] = extract_values([s[0].strip() for s in series])
    result['y_values'] = extract_values([s[1].strip() for s in series])

    return result

def parse_info_for_line_chart(chart_info):
    result = {}
    categories = chart_info[0].split('|')
    x_label = categories[0].strip()
    categories = [cat.strip() for cat in categories[1:]]
    result['x_label'] = x_label
    result['categories'] = categories
    series = [chart_info[i].split('|') for i in range(1,len(chart_info))]
    result['x_values'] = [x[0].strip() for x in series]
    result['series'] = {categories[i]: extract_values([s[i+1].strip() for s in series]) for i in range(len(categories))}

    return result


def get_chart_info(image_path, nearby_text=""):
    chart_info = {}
    chart_structure = get_chart_structure(image_path)
    chart_structure = chart_structure.split('\n')

    chart_info['data_table'] = chart_structure
    chart_info['file_path'] = image_path
    chart_info['chart_type'] = get_chart_type(image_path)
    chart_info['nearby_text'] = nearby_text

    try:
        chart_info['chart_title'] = chart_structure[0].split('|')[1].strip()
    except (IndexError, AttributeError):
        chart_info['chart_title'] = "unavailable"

    try:
        if chart_info['chart_type'] == 'Pie Chart':
            chart_info.update(parse_info_for_pie_chart(chart_structure[1:]))
        elif chart_info['chart_type'] == 'Bar Chart':
            chart_info.update(parse_info_for_bar_chart(chart_structure[1:]))
        elif chart_info['chart_type'] == 'Scatter Plot':
            chart_info.update(parse_info_for_scatter_chart(chart_structure[1:]))
        elif chart_info['chart_type'] == 'Line Chart':
            chart_info.update(parse_info_for_line_chart(chart_structure[1:]))
        else:
            chart_info.update(parse_info_general_fixed(chart_structure[1:]))
    except:
        print("Something went wrong when parsing chart info")
        return chart_info

    return chart_info

def get_empty_text_info():
    """Return empty text info structure"""
    return {
        "title": "unavailable",
        "full_text": "unavailable",
    }


def extract_vectorial_drawings(page, output_dir, page_number, global_counter):
    """Extract only the first/largest vectorial drawing from the page"""
    drawings = []

    # Get all drawings (paths, lines, shapes) from the page
    paths = page.get_drawings()

    if paths:
        # Group nearby paths to identify complete drawings
        drawing_groups = group_nearby_paths(paths)

        # Filter groups by size and select the largest one
        valid_groups = []
        for group in drawing_groups:
            bbox = calculate_group_bbox(group)
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height

                # Only consider drawings with meaningful size
                if width > 50 and height > 50 and area > 2500:
                    valid_groups.append((group, bbox, area))

        if valid_groups:
            # Sort by area (largest first) and take only the first one
            valid_groups.sort(key=lambda x: x[2], reverse=True)
            path_group, bbox, area = valid_groups[0]

            # Expand bbox with minimal padding for context
            expanded_bbox = fitz.Rect(
                max(0, bbox[0] - 3),
                max(0, bbox[1] - 3),
                min(page.rect.width, bbox[2] + 3),
                min(page.rect.height, bbox[3] + 3)
            )

            # Render this region as image
            mat = fitz.Matrix(2, 2)  # 2x scale for better quality
            pix = page.get_pixmap(matrix=mat, clip=expanded_bbox)

            # Save the drawing as image
            drawing_filename = f"page_{page_number + 1}_drawing_1.png"
            drawing_path = os.path.join(output_dir, drawing_filename)
            pix.save(drawing_path)

            # Extract nearby text - exclude text inside the drawing area
            nearby_text = extract_text_outside_area(page, bbox)

            # Use get_chart_info function instead of manual JSON building
            chart_info = get_chart_info(drawing_path, nearby_text)

            # Add metadata specific to vectorial drawings using global counter
            chart_info.update({
                "page_number": page_number + 1,
                "image_number": global_counter,
            })

            drawings.append(chart_info)

            pix = None  # Free memory

    return drawings


def extract_text_outside_area(page, drawing_bbox, padding=50):
    """Extract text near the drawing but not inside it"""
    # Create expanded area for text search
    search_area = fitz.Rect(
        max(0, drawing_bbox[0] - padding),
        max(0, drawing_bbox[1] - padding),
        min(page.rect.width, drawing_bbox[2] + padding),
        min(page.rect.height, drawing_bbox[3] + padding)
    )

    # Get all text blocks in the search area
    text_blocks = page.get_text("dict", clip=search_area)

    nearby_text_parts = []
    drawing_rect = fitz.Rect(drawing_bbox[0], drawing_bbox[1], drawing_bbox[2], drawing_bbox[3])

    # Filter out text that overlaps with the drawing area
    for block in text_blocks.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    span_rect = fitz.Rect(span["bbox"])

                    # Only include text that doesn't overlap with the drawing
                    if not span_rect.intersects(drawing_rect):
                        text = span["text"].strip()
                        if text:
                            nearby_text_parts.append(text)

    return " ".join(nearby_text_parts).strip()


def group_nearby_paths(paths, distance_threshold=30):
    """Group paths that are close together to form complete drawings"""
    if not paths:
        return []

    groups = []
    used_paths = set()

    for i, path in enumerate(paths):
        if i in used_paths:
            continue

        # Start a new group
        current_group = [path]
        used_paths.add(i)

        # Find nearby paths
        path_bbox = calculate_path_bbox(path)
        if not path_bbox:
            continue

        for j, other_path in enumerate(paths):
            if j in used_paths:
                continue

            other_bbox = calculate_path_bbox(other_path)
            if not other_bbox:
                continue

            # Check if paths are close enough
            if paths_are_nearby(path_bbox, other_bbox, distance_threshold):
                current_group.append(other_path)
                used_paths.add(j)

        if current_group:
            groups.append(current_group)

    return groups


def calculate_path_bbox(path):
    """Calculate bounding box for a single path"""
    try:
        if 'rect' in path:
            rect = path['rect']
            return (rect.x0, rect.y0, rect.x1, rect.y1)
        elif 'items' in path:
            all_points = []
            for item in path['items']:
                if item[0] in ['l', 'm', 'c']:  # line, move, curve
                    all_points.extend(item[1:])

            if all_points:
                x_coords = [p.x if hasattr(p, 'x') else p[0] for p in all_points]
                y_coords = [p.y if hasattr(p, 'y') else p[1] for p in all_points]
                return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    except:
        pass
    return None


def calculate_group_bbox(path_group):
    """Calculate combined bounding box for a group of paths"""
    bboxes = [calculate_path_bbox(path) for path in path_group]
    bboxes = [bbox for bbox in bboxes if bbox]

    if not bboxes:
        return None

    min_x = min(bbox[0] for bbox in bboxes)
    min_y = min(bbox[1] for bbox in bboxes)
    max_x = max(bbox[2] for bbox in bboxes)
    max_y = max(bbox[3] for bbox in bboxes)

    return (min_x, min_y, max_x, max_y)


def paths_are_nearby(bbox1, bbox2, threshold):
    """Check if two bounding boxes are within threshold distance"""
    # Calculate distance between box centers
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2

    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    return distance <= threshold and distance != 0


def get_image_regions(pdf_path, output_dir="extracted_images"):
    """Extract all embedded images and the first vectorial drawing from PDF"""
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print("Processing PDF")

    pdf_doc = fitz.open(pdf_path)
    results = []
    global_image_counter = 0  # Global counter for all images and drawings

    for page_number in range(len(pdf_doc)):
        page = pdf_doc[page_number]

        # Extract embedded images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]

            try:
                # Extract image data from PDF
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Increment global counter
                global_image_counter += 1

                # Save image with appropriate extension using global counter
                img_filename = f"page_{page_number + 1}_image_{global_image_counter}.{image_ext}"
                img_path = os.path.join(output_dir, img_filename)

                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)

                # Get image rectangle on page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]

                    # Convert rect to bbox format for consistency with vectorial drawings
                    bbox = (rect.x0, rect.y0, rect.x1, rect.y1)

                    # Extract nearby text - exclude text inside the image area (same logic as vectorial drawings)
                    nearby_text = extract_text_outside_area(page, bbox)

                    # Use get_chart_info function instead of manual JSON building
                    chart_info = get_chart_info(img_path, nearby_text)

                    # Add metadata specific to embedded images using global counter
                    chart_info.update({
                        "page_number": page_number + 1,
                        "image_number": global_image_counter,
                    })

                    results.append(chart_info)

            except Exception as e:
                print(f"Error extracting image {img_index + 1} from page {page_number + 1}: {e}")
                continue

        # Extract only the first/largest vectorial drawing
        try:
            drawings = extract_vectorial_drawings(page, output_dir, page_number, global_image_counter + 1)
            results.extend(drawings)
            # Update global counter if drawings were found
            if drawings:
                global_image_counter += len(drawings)
        except Exception as e:
            print(f"Error extracting drawings from page {page_number + 1}: {e}")

    pdf_doc.close()

    # Save JSON file
    json_path = os.path.join(output_dir, "extraction_results.json")
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2, ensure_ascii=False)

    return results

def convert_docx_to_pdf(input_path, output_path):
    """Convert a .docx file to PDF using Microsoft Word COM API"""
    try:
        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(input_path))
        doc.SaveAs(os.path.abspath(output_path), FileFormat=17)  # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        return True
    except Exception as e:
        print(f"Error converting .docx to PDF: {e}")
        return False
    finally:
        if 'word' in locals():
            word.Quit()


def convert_word_to_pdf(file_path, output_path):
    """Convert a .docx file to PDF using docx2pdf library"""
    try:
        # Check if output_path has .pdf extension, if not add it
        if not output_path.lower().endswith('.pdf'):
            output_path = os.path.splitext(output_path)[0] + '.pdf'

        # Convert the document
        convert(file_path, output_path)

        # Verify the PDF was created
        if os.path.exists(output_path):
            return output_path
        else:
            print(f"PDF conversion completed but file not found at: {output_path}")
            return None
    except Exception as e:
        print(f"Error converting .docx to PDF using docx2pdf: {e}")
        return None

def convert_pptx_to_pdf(input_path, output_path):
    """Convert a .pptx file to PDF using Microsoft PowerPoint COM API"""
    try:
        powerpoint = comtypes.client.CreateObject('PowerPoint.Application')
        powerpoint.Visible = False
        presentation = powerpoint.Presentations.Open(os.path.abspath(input_path))
        presentation.SaveAs(os.path.abspath(output_path), FileFormat=32)  # 32 = ppSaveAsPDF
        presentation.Close()
        powerpoint.Quit()
        return True
    except Exception as e:
        print(f"Error converting .pptx to PDF: {e}")
        return False
    finally:
        if 'powerpoint' in locals():
            powerpoint.Quit()


from pathlib import Path
import os


def convert_to_pdf_if_needed(file_path, temp_dir="temp_pdf"):
    """Convert .docx or .pptx to PDF if needed, return path to PDF file"""
    Path(temp_dir).mkdir(exist_ok=True)
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext in ['.docx', '.pptx']:
        temp_pdf_path = os.path.join(temp_dir, f"converted_{os.path.splitext(os.path.basename(file_path))[0]}.pdf")

        try:
            if file_ext == '.docx':
                # docx2pdf returns path; LibreOffice version may also return path
                pdf_path = convert_word_to_pdf(file_path, file_path)
            # else:  # .pptx
            #     pdf_path = convert_pptx_to_pdf(file_path, temp_pdf_path)

            if pdf_path and os.path.exists(pdf_path):
                return pdf_path
            else:
                raise ValueError(f"Conversion failed: {file_ext} → PDF")
        except Exception as e:
            raise ValueError(f"Error converting {file_ext} to PDF: {e}")

    elif file_ext == '.pdf':
        return file_path
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")


def parse_file(file_path):
    if os.path.exists(file_path):
        try:
            pdf_path = convert_to_pdf_if_needed(file_path)
            results = get_image_regions(pdf_path)
            # Clean up temporary PDF if it was created
            if file_path != pdf_path:
                try:
                    os.remove(pdf_path)
                except:
                    pass
            return results
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    else:
        print(f"File not found: {file_path}")
        return []

def main():
    pdf_path = "../sample_data/test_pt_demo(7).pdf"

    if os.path.exists(pdf_path):
        results = parse_file(pdf_path)

        # Separate results by type
        embedded_images = [r for r in results if r.get("type") == "embedded_image"]
        vectorial_drawings = [r for r in results if r.get("type") == "vectorial_drawing"]

        print(f"Extracted {len(embedded_images)} embedded images")
        print(f"Extracted {len(vectorial_drawings)} vectorial drawings (first/largest only)")
        print("Results saved in 'extracted_images' directory")
    # results = get_chart_info("chart_images_test/line-chart-example2.png")
    # output_dir = "extracted_images"
    # json_path = os.path.join(output_dir, "extraction_results.json")
    # with open(json_path, "w", encoding="utf-8") as json_file:
    #     json.dump(results, json_file, indent=2, ensure_ascii=False)








if __name__ == "__main__":
    main()