import fitz  # PyMuPDF
import os
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2


def detect_chart_regions(image_path, min_area=10000):
    """
    Detect potential chart regions in an image using contour detection
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and aspect ratio
    chart_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Charts typically have reasonable aspect ratios (not too thin/tall)
            if 0.3 < aspect_ratio < 5.0 and w > 100 and h > 100:
                # Add some padding around the detected region
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2 * padding)
                h = min(img.shape[0] - y, h + 2 * padding)

                chart_regions.append((x, y, w, h))

    # Remove overlapping regions (keep the larger one)
    filtered_regions = []
    for i, (x1, y1, w1, h1) in enumerate(chart_regions):
        is_overlapping = False
        for j, (x2, y2, w2, h2) in enumerate(chart_regions):
            if i != j:
                # Check if regions overlap significantly
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                area1 = w1 * h1
                area2 = w2 * h2

                if overlap_area > 0.5 * min(area1, area2):
                    # Keep the larger region
                    if area1 < area2:
                        is_overlapping = True
                        break

        if not is_overlapping:
            filtered_regions.append((x1, y1, w1, h1))

    return filtered_regions


def is_likely_chart(image_path):
    """
    Determine if an image is likely to contain a chart based on various heuristics
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img.convert('RGB'))

        # Check image size (charts are usually reasonably sized)
        if img.width < 100 or img.height < 100:
            return False

        # Check for sufficient color variation (charts usually have multiple colors)
        std_dev = np.std(img_array)
        if std_dev < 15:
            return False

        # Check aspect ratio (charts have reasonable aspect ratios)
        aspect_ratio = img.width / img.height
        if aspect_ratio < 0.2 or aspect_ratio > 10:
            return False

        # Additional check: look for line-like structures using edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])

        # Charts typically have a good amount of edges (lines, borders, etc.)
        if edge_ratio > 0.01:
            return True

        return False
    except Exception as e:
        print(f"Error checking if image is chart: {e}")
        return False


def extract_charts_from_pdf(pdf_path, output_dir="output_charts", dpi=300):
    """
    Extract individual charts from PDF as separate PNG files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    chart_count = 0

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        print(f"Processing page {page_num + 1}...")

        # First, try to extract embedded images (charts stored as images)
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Save temporary image
                temp_filename = os.path.join(output_dir, f"temp_img_page{page_num + 1}_{img_index}.png")
                with open(temp_filename, "wb") as image_file:
                    image_file.write(image_bytes)

                # Check if it's likely a chart
                if is_likely_chart(temp_filename):
                    chart_count += 1
                    final_filename = os.path.join(output_dir, f"chart_{chart_count}.png")
                    os.rename(temp_filename, final_filename)
                    print(f"Extracted embedded chart: {final_filename}")
                else:
                    os.remove(temp_filename)

            except Exception as e:
                print(f"Error extracting embedded image: {e}")

        # Render page as high-quality image to detect vector-based charts
        pix = page.get_pixmap(dpi=dpi)
        page_image_path = os.path.join(output_dir, f"temp_page_{page_num + 1}.png")
        pix.save(page_image_path)

        # Detect chart regions in the full page image
        chart_regions = detect_chart_regions(page_image_path)

        if chart_regions:
            # Load the page image
            page_img = Image.open(page_image_path)

            for region_index, (x, y, w, h) in enumerate(chart_regions):
                # Crop the chart region
                chart_region = page_img.crop((x, y, x + w, y + h))

                # Save cropped chart region as temporary file
                temp_chart_path = os.path.join(output_dir, f"temp_chart_page{page_num + 1}_region{region_index}.png")
                chart_region.save(temp_chart_path)

                # Verify it's actually a chart
                if is_likely_chart(temp_chart_path):
                    chart_count += 1
                    final_chart_path = os.path.join(output_dir, f"chart_{chart_count}.png")
                    os.rename(temp_chart_path, final_chart_path)
                    print(f"Extracted chart region: {final_chart_path}")
                else:
                    os.remove(temp_chart_path)

        # Clean up temporary page image
        if os.path.exists(page_image_path):
            os.remove(page_image_path)

    pdf_document.close()
    print(f"Extraction complete. Found {chart_count} charts.")
    return chart_count


if __name__ == "__main__":
    # Example usage
    pdf_file = "../sample_data/MHP Statistics.pdf"  # Replace with your PDF file path
    extract_charts_from_pdf(pdf_file)