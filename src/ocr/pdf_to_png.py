import os
import glob
from pdf2image import convert_from_path
import re
import time

BASE_IMAGE_DIR = "/data/images"
PDF_DIRECTORIES = [
    "/data/pds/sources", # Example active directory
    # Add other default directories here if needed, or leave empty to rely on CLI args
]

def convert_pdf_to_images(pdf_path, output_dir, paper_name):
    """
    Convert a PDF to PNG images, one per page.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Base directory to save images
        paper_name (str): Name of the paper (used for folder name)
    
    Returns:
        list: List of paths to the generated images, or empty list if conversion fails
    """
    start_time = time.time()
    print(f"Converting PDF: {os.path.basename(pdf_path)}")
    
    paper_dir = os.path.join(output_dir, paper_name)
    os.makedirs(paper_dir, exist_ok=True)
    
    print("  Generating images from PDF...")
    pdf_start_time = time.time()
    try:
        images = convert_from_path(pdf_path, dpi=300)
        pdf_end_time = time.time()
        print(f"  PDF conversion completed in {pdf_end_time - pdf_start_time:.2f} seconds")
        print(f"  Total pages: {len(images)}")
        
        image_paths = []
        print("  Saving pages as PNG files...")
        for i, image in enumerate(images):
            page_num = i + 1
            if len(images) > 10 and page_num % 5 == 0:
                print(f"    Progress: {page_num}/{len(images)} pages")
            output_path = os.path.join(paper_dir, f"page_{page_num}.png")
            image.save(output_path, "PNG")
            image_paths.append(output_path)
        
        end_time = time.time()
        print(f"  Finished saving {len(images)} pages in {end_time - start_time:.2f} seconds")
        return image_paths
    except Exception as e:
        print(f"  ERROR: Failed to convert PDF: {os.path.basename(pdf_path)}")
        print(f"  Error details: {str(e)}")
        return []

def process_pdf_directory(pdf_dir, output_base_dir):
    """
    Process all PDFs in a directory, converting each to PNGs.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        output_base_dir (str): Base directory to save images
    
    Returns:
        dict: Dictionary mapping paper names to lists of image paths
        list: List of skipped files
        list: List of already processed files
    """
    start_time = time.time()
    result = {}
    skipped_files = []
    already_processed = []
    
    folder_name = os.path.basename(os.path.normpath(pdf_dir))
    output_dir = os.path.join(output_base_dir, folder_name)
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process in {pdf_dir}")
    
    for idx, pdf_file in enumerate(pdf_files):
        print(f"\nProcessing PDF {idx+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
        pdf_process_start_time = time.time()
        
        paper_name_original = os.path.splitext(os.path.basename(pdf_file))[0]
        
        paper_output_dir_check = os.path.join(output_dir, paper_name_original) # Check with original name first
        if os.path.exists(paper_output_dir_check) and os.path.isdir(paper_output_dir_check):
            png_files = glob.glob(os.path.join(paper_output_dir_check, "*.png"))
            if png_files:
                print(f"Skipping PDF: Image folder already exists for {paper_name_original}")
                already_processed.append(os.path.basename(pdf_file))
                continue
        
        try:
            # Normalize paper_name for folder creation and internal use after initial check
            # 1. Remove spaces adjacent to hyphens
            paper_name_normalized = paper_name_original.replace(" -", "-").replace("- ", "-")
            # 2. Replace remaining spaces with underscores
            paper_name_normalized = paper_name_normalized.replace(" ", "_")
            # 3. Remove any special characters not in {a-zA-Z0-9-_}
            paper_name_normalized = re.sub(r'[^a-zA-Z0-9\-_]', '', paper_name_normalized)

            image_paths = convert_pdf_to_images(pdf_file, output_dir, paper_name_normalized)
            
            if not image_paths:
                print(f"Skipping PDF due to conversion errors: {os.path.basename(pdf_file)}")
                skipped_files.append(os.path.basename(pdf_file))
                continue
            
            result[paper_name_normalized] = image_paths # Use normalized name as key
            
            pdf_process_end_time = time.time()
            print(f"Completed PDF {idx+1}/{len(pdf_files)} in {pdf_process_end_time - pdf_process_start_time:.2f} seconds")
        except Exception as e:
            print(f"ERROR: Failed to process PDF: {os.path.basename(pdf_file)}")
            print(f"Error details: {str(e)}")
            skipped_files.append(os.path.basename(pdf_file))
        
    end_time = time.time()
    print(f"\nTotal processing time for all PDFs in {folder_name}: {end_time - start_time:.2f} seconds")
    
    if already_processed:
        print(f"\nSkipped {len(already_processed)} already processed PDFs in {folder_name}:")
        for file in already_processed:
            print(f"  - {file}")
    
    if skipped_files:
        print(f"\nSkipped {len(skipped_files)} problematic PDF files in {folder_name}:")
        for file in skipped_files:
            print(f"  - {file}")
    
    return result, skipped_files, already_processed

def main():
    """
    Main function to convert PDFs to PNGs from multiple directories.
    """
    overall_start_time = time.time()
    
    current_pdf_directories = PDF_DIRECTORIES
    if len(sys.argv) > 1:
        # If directories are provided as command-line arguments, use those
        current_pdf_directories = sys.argv[1:]
        print(f"Using PDF directories from command line: {current_pdf_directories}")
    elif not PDF_DIRECTORIES:
        print("No PDF directories specified in script or command line. Exiting.")
        return

    print(f"Processing PDFs from {len(current_pdf_directories)} directories")
    
    all_results = {}
    all_skipped = []
    all_already_processed = []
    
    for pdf_dir in current_pdf_directories:
        print(f"\n{'='*50}")
        print(f"Processing directory: {pdf_dir}")
        print(f"{'='*50}")
        
        if not os.path.isdir(pdf_dir):
            print(f"Warning: Directory {pdf_dir} does not exist or is not a directory. Skipping.")
            all_skipped.append((pdf_dir, "Directory not found"))
            continue
            
        result, skipped, already_processed_in_dir = process_pdf_directory(pdf_dir, BASE_IMAGE_DIR)
        
        all_results.update(result)
        all_skipped.extend([(pdf_dir, file) for file in skipped])
        all_already_processed.extend([(pdf_dir, file) for file in already_processed_in_dir])
    
    print(f"\n{'='*50}")
    print("OVERALL SUMMARY")
    print(f"{'='*50}")
    print(f"Processed {len(current_pdf_directories)} directories")
    print(f"Successfully converted/found images for {len(all_results)} PDFs")
    
    total_pages = 0
    for paper_name, image_paths in all_results.items():
        pages = len(image_paths)
        total_pages += pages
    
    print(f"Total pages existing/converted: {total_pages}")
    
    if all_already_processed:
        print(f"\nTotal already processed PDFs (skipped during run): {len(all_already_processed)}")

        
    if all_skipped:
        print(f"\nTotal problematic PDFs/directories (skipped): {len(all_skipped)}")
        print("Skipped items by directory:")
        for dir_path, file_or_reason in all_skipped:
            print(f"  - Directory: {os.path.basename(dir_path)}, Item/Reason: {file_or_reason}")
    
    overall_end_time = time.time()
    print(f"\nTotal execution time: {overall_end_time - overall_start_time:.2f} seconds")
    
    return all_results

if __name__ == "__main__":
    import sys
    main()
