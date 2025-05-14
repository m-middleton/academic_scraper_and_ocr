import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import requests
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string
import time
import itertools
import datetime

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


 
translation_table = str.maketrans(punctuation_dict)


class TimeoutStoppingCriteria(StoppingCriteria):
    """A stopping criteria that stops generation if a timeout is reached."""
    def __init__(self, timeout_seconds=15):
        """Initializes the TimeoutStoppingCriteria.

        Args:
            timeout_seconds (int): The timeout duration in seconds.
        """
        super().__init__() # Added super call
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout_seconds

    @property
    def name(self):
        return "TimeoutStoppingCriteria"


def load_image(image_file: str) -> Image.Image:
    """Loads an image from a file path or URL.

    Args:
        image_file (str): The path or URL to the image.

    Returns:
        Image.Image: The loaded image object in RGB format.
    """
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_model_and_tokenizer(model_name: str):
    """
    Load and return the model and tokenizer for OCR.
    
    Args:
        model_name: Path or name of the model to load
        
    Returns:
        tokenizer: The loaded tokenizer
        model: The loaded model
    """
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = GOTQwenForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        device_map='cuda', 
        use_safetensors=True, 
        pad_token_id=151643 # Specific to Qwen models
    ).eval()
    
    model.to(device='cuda', dtype=torch.bfloat16)
    
    return tokenizer, model


def prepare_ocr_prompt(image_path, ocr_type='text', box=None, color=None):
    """
    Prepare the OCR prompt for the given image and parameters.
    
    Args:
        image_path: Path to the image file
        ocr_type: Type of OCR, 'text' or 'format'
        box: Bounding box coordinates
        color: Color specification
        
    Returns:
        image: The loaded image
        prompt: The prepared prompt
        qs: The query string
        use_im_start_end: Whether to use image start/end tokens
        image_token_len: Length of image token sequence
    """
    image_processor = BlipImageEvalProcessor(image_size=1024)
    image_processor_high = BlipImageEvalProcessor(image_size=1024)
    
    use_im_start_end = True
    image_token_len = 256
    
    image = load_image(image_path)
    w, h = image.size
    
    if ocr_type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '
    
    if box:
        bbox = eval(box) # Potential security risk with eval if box comes from untrusted source
        # Bounding box coordinates are scaled to a 1000x1000 grid
        if len(bbox) == 2:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
            bbox[2] = int(bbox[2]/w*1000)
            bbox[3] = int(bbox[3]/h*1000)
        if ocr_type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '
    
    if color:
        if ocr_type == 'format':
            qs = '[' + color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + color + ']' + ' ' + 'OCR: '
    
    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    return image, qs, image_processor, image_processor_high


def generate_ocr_output(model, tokenizer, image, qs, image_processor, image_processor_high, timeout_seconds=15):
    """
    Generate OCR output for the given image and prompt.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        image: The image to process
        qs: The query string
        image_processor: The image processor
        image_processor_high: The high-resolution image processor
        timeout_seconds: Maximum time in seconds for generation before timing out
        
    Returns:
        output_text: The OCR output text
        output_ids: The raw output IDs
        skip_reason: Reason for skipping if applicable, None otherwise
    """
    conv_mode = "mpt"
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print(prompt) # This print might be for debugging, consider removing for production
    
    inputs = tokenizer([prompt])
    
    image_1 = image.copy()
    image_tensor = image_processor(image)
    image_tensor_1 = image_processor_high(image_1)
    
    input_ids = torch.as_tensor(inputs.input_ids).cuda()
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    keyword_stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    timeout_stopping_criteria = TimeoutStoppingCriteria(timeout_seconds=timeout_seconds)
    
    skip_reason = None
    outputs = "" # Initialize outputs
    output_ids = None # Initialize output_ids
    
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids,
                images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=20,
                max_new_tokens=4096,
                stopping_criteria=[keyword_stopping_criteria, timeout_stopping_criteria]
            )
            
            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
    
    except Exception as e:
        error_msg = str(e)
        print(f"Generation error: {error_msg}")
        outputs = f"Error during generation: {error_msg}"
        # output_ids remains None as initialized
        skip_reason = f"Generation error: {error_msg}"
        
    if time.time() - timeout_stopping_criteria.start_time > timeout_seconds and not skip_reason: # Check if not already skipped due to other error
        print(f"Generation timed out after {timeout_seconds} seconds")
        if not outputs: # If no output was generated at all before timeout
            outputs = "Generation timed out"
        else: # If partial output exists
            outputs = outputs + " [TIMED OUT]"
        skip_reason = "Generation timed out"
        
    return outputs, output_ids, stop_str, conv, skip_reason


def render_ocr_output(outputs, stop_str, ocr_type):
    """
    Render the OCR output as HTML.
    
    Args:
        outputs: The OCR output text
        stop_str: The stop string
        ocr_type: Type of OCR, 'text' or 'format'
        
    Returns:
        None (writes to file)
    """
    print('==============rendering===============') # This print might be for debugging
    
    if '**kern' in outputs:
        import verovio
        from cairosvg import svg2png
        import cv2
        import numpy as np
        tk = verovio.toolkit()
        tk.loadData(outputs)
        tk.setOptions({"pageWidth": 2100, "footer": 'none',
        'barLineWidth': 0.5, 'beamMaxSlope': 15,
        'staffLineWidth': 0.2, 'spacingStaff': 6})
        tk.getPageCount()
        svg = tk.renderToSVG()
        svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")

        svg_to_html(svg, "./results/demo.html")

    if ocr_type == 'format' and '**kern' not in outputs:
        
        if  '\\begin{tikzpicture}' not in outputs:
            html_path = "./render_tools/" + "/content-mmd-to-html.html"
            html_path_2 = "./results/demo.html"
            right_num = outputs.count('\\right')
            left_num = outputs.count('\left')

            if right_num != left_num:
                outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace('\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace('\\right|', '|').replace('\left.', '.').replace('\\right.', '.')

            outputs = outputs.replace('"', '``').replace('$', '')

            outputs_list = outputs.split('\n')
            gt= ''
            for out in outputs_list:
                gt +=  '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n' 
            
            gt = gt[:-2]

            with open(html_path, 'r') as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + 'const text ='  + gt  + lines[1]
        else:
            html_path = "./render_tools/" + "/tikz.html"
            html_path_2 = "./results/demo.html"
            outputs = outputs.translate(translation_table)
            outputs_list = outputs.split('\n')
            gt= ''
            for out in outputs_list:
                if out:
                    if '\\begin{tikzpicture}' not in out and '\\end{tikzpicture}' not in out:
                        while out[-1] == ' ':
                            out = out[:-1]
                            if out is None:
                                break

                        if out:
                            if out[-1] != ';':
                                gt += out[:-1] + ';\n'
                            else:
                                gt += out + '\n'
                    else:
                        gt += out + '\n'

            with open(html_path, 'r') as web_f:
                lines = web_f.read()
                lines = lines.split("const text =")
                new_web = lines[0] + gt + lines[1]

        with open(html_path_2, 'w') as web_f_new:
            web_f_new.write(new_web)


def process_image_ocr(model, tokenizer, image_path, ocr_type='text', box=None, color=None, render=False, timeout_seconds=15):
    """
    Process an image with OCR and return the extracted text.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        image_path: Path to the image file
        ocr_type: Type of OCR, 'text' or 'format'
        box: Bounding box coordinates as string
        color: Color specification
        render: Whether to render the output as HTML
        timeout_seconds: Maximum time in seconds for generation before timing out
        
    Returns:
        output_text: The OCR output text
        skip_reason: Reason for skipping if applicable, None otherwise
    """
    image, qs, image_processor, image_processor_high = prepare_ocr_prompt(image_path, ocr_type, box, color)
    outputs, output_ids, stop_str, conv, skip_reason = generate_ocr_output(
        model, 
        tokenizer, 
        image, 
        qs, 
        image_processor, 
        image_processor_high, 
        timeout_seconds=timeout_seconds
    )
    
    if render and output_ids is not None and skip_reason is None:
        render_ocr_output(outputs, stop_str, ocr_type)
    
    return outputs, skip_reason

def get_paper_images(base_input, groups, max_images=50):
    """
    Scans the images directory to find all papers and their page images for specified groups.
    Directory structure: <base_input>/<group_name>/<single_paper>/<multiple png images>
    
    Args:
        base_input: Base input directory
        groups: List of group names to process
        max_images: Maximum number of images allowed per paper (papers with more will be skipped)
        
    Returns:
        dict: Dictionary mapping (group_name, paper_name) to lists of image paths
        list: List of skipped papers due to too many images or filename errors
    """
    result = {}
    skipped_papers = []
    
    for group_name in groups:
        group_dir = os.path.join(base_input, group_name)
        
        if not os.path.isdir(group_dir):
            print(f"Warning: Group directory {group_dir} not found or not a directory. Skipping.")
            continue
        
        for paper_name in os.listdir(group_dir):
            paper_dir = os.path.join(group_dir, paper_name)
            
            if not os.path.isdir(paper_dir):
                continue
            
            image_files = [
                os.path.join(paper_dir, filename)
                for filename in os.listdir(paper_dir)
                if filename.lower().endswith('.png')
            ]
            
            if not image_files:
                print(f"Skipping {group_name}/{paper_name}: No PNG images found in the directory")
                continue
            
            if len(image_files) > max_images:
                skip_reason = f"Too many images ({len(image_files)} > {max_images})"
                print(f"Skipping {group_name}/{paper_name}: {skip_reason}")
                skipped_papers.append((group_name, paper_name, skip_reason))
                continue
            
            # Extract page number from filename and sort numerically
            try:
                image_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            except (IndexError, ValueError) as e:
                print(f"Warning: Could not sort image files for {group_name}/{paper_name} due to unexpected filename format. Error: {e}. Skipping paper.")
                skipped_papers.append((group_name, paper_name, f"Filename format error: {e}"))
                continue

            result[(group_name, paper_name)] = image_files
    
    return result, skipped_papers

def log_skipped_papers(base_output, skipped_papers, skipped_pages=None):
    """
    Log skipped papers and pages to a text file.
    
    Args:
        base_output: Base output directory
        skipped_papers: List of (group_name, paper_name, reason) tuples for skipped papers.
                        Reasons could be too many pages or filename format errors.
        skipped_pages: List of (group_name, paper_name, image_path, reason) tuples for skipped pages.
                       Reasons could be processing errors like timeout or generation failure.
    """
    if not skipped_papers and not skipped_pages:
        return
    
    log_file = os.path.join(base_output, "skipped_log.txt") 
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a") as f:
        f.write(f"--- Log entry {timestamp} ---\n")
        
        if skipped_papers:
            f.write("== Skipped Papers (due to file count/format issues) ==\n")
            for group_name, paper_name, reason in skipped_papers:
                f.write(f"{group_name}/{paper_name}: {reason}\n")
            f.write("\n")
        
        if skipped_pages:
            f.write("== Skipped Pages (due to processing errors) ==\n")
            for group_name, paper_name, image_path, reason in skipped_pages:
                page_image_filename = os.path.basename(image_path)
                page_number_str = "UnknownPage"
                try:
                    # Assuming filename format like 'page_001.png' or similar
                    page_number_str = os.path.splitext(page_image_filename)[0].split('_')[-1]
                except IndexError:
                    pass # Keep page_number_str as UnknownPage
                f.write(f"{group_name}/{paper_name} - Page Image: {page_image_filename} (Num: {page_number_str}): {reason}\n")
            f.write("\n")
    
    if skipped_papers:
        print(f"Logged {len(skipped_papers)} skipped papers to {log_file}")
    if skipped_pages:
        print(f"Logged {len(skipped_pages)} skipped pages to {log_file}")

def save_processed_text(processed_text, output_file):
    """
    Save the processed text to a file.
    
    Args:
        processed_text: The processed text to save
        output_file: The file to save the processed text to
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(processed_text)

def process_papers_interleaved(model, tokenizer, papers_dict, base_output_folder, ocr_type='text', box=None, color=None, render=False, timeout_seconds=15):
    """
    Process papers in an interleaved manner - one paper from each group at a time.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        papers_dict: Dictionary mapping (group_name, paper_name) to image file lists
        base_output_folder: Base output directory
        ocr_type: Type of OCR
        box: Bounding box coordinates
        color: Color specification
        render: Whether to render output as HTML
        timeout_seconds: Maximum time in seconds for generation before timing out
    Returns:
        Tuple[List[str], List[Tuple[str,str,str,str]]]: 
            - List of paths to successfully processed paper text files.
            - List of (group_name, paper_name, image_path, reason) for skipped pages.
    """
    if not papers_dict:
        print("No papers to process.")
        return [], [] 
    
    papers_by_group = {}
    for (group_name, paper_name), image_files in papers_dict.items():
        if group_name not in papers_by_group:
            papers_by_group[group_name] = []
        papers_by_group[group_name].append((paper_name, image_files))
    
    group_names = list(papers_by_group.keys())
    max_papers_in_any_group = max(len(papers_by_group[group]) for group in group_names) if group_names else 0
    
    total_papers_to_attempt = sum(len(papers_by_group[group]) for group in group_names)
    processed_papers_count = 0
    already_existed_count = 0
    
    all_successfully_processed_outputs = []
    all_skipped_pages_accumulator = []

    for paper_idx in range(max_papers_in_any_group):
        for group_name in group_names:
            if paper_idx >= len(papers_by_group[group_name]):
                continue
            
            paper_name, image_files = papers_by_group[group_name][paper_idx]
            
            group_output_dir = os.path.join(base_output_folder, group_name)
            os.makedirs(group_output_dir, exist_ok=True)
            
            output_file = os.path.join(group_output_dir, f"{paper_name}.txt")
            if os.path.exists(output_file):
                print(f"Skipping {group_name}/{paper_name}: Output file already exists ({output_file})")
                already_existed_count += 1
                processed_papers_count +=1 # Count as processed for progress tracking
                all_successfully_processed_outputs.append(output_file) # Consider it successfully processed if it exists
                continue
            
            processed_papers_count += 1
            remaining_papers = total_papers_to_attempt - processed_papers_count
            
            print(f"Processing {group_name}/{paper_name}... ({processed_papers_count}/{total_papers_to_attempt}, {remaining_papers} remaining)")
            
            current_paper_text = ""
            num_pages = len(image_files)
            total_time_for_paper = 0
            paper_specific_skipped_pages = []
            
            for page_number, image_file in enumerate(image_files):
                start_time = time.time()
                single_page_text, skip_reason = process_image_ocr(
                    model,
                    tokenizer,
                    image_file,
                    ocr_type=ocr_type,
                    box=box,
                    color=color,
                    render=render,
                    timeout_seconds=timeout_seconds
                )
                current_paper_text += f'{single_page_text}\n' # Append even if there was a skip_reason, text might be partial or error msg
                
                end_time = time.time()
                page_time = end_time - start_time
                total_time_for_paper += page_time
                
                if skip_reason:
                    print(f"\t Page {page_number+1} of {num_pages}: {page_time:.2f} seconds - SKIPPED: {skip_reason}")
                    # Use image_file (full path) for logging skipped pages
                    paper_specific_skipped_pages.append((group_name, paper_name, image_file, skip_reason))
                else:
                    print(f"\t Page {page_number+1} of {num_pages}: {page_time:.2f} seconds; total time: {total_time_for_paper:.2f} seconds")
            
            save_processed_text(current_paper_text.strip(), output_file) # Save stripped text
            all_successfully_processed_outputs.append(output_file)
            print(f"Generated text for {group_name}/{paper_name} saved to {output_file}")
            
            if paper_specific_skipped_pages:
                all_skipped_pages_accumulator.extend(paper_specific_skipped_pages)
                # Log skipped pages for this paper immediately
                log_skipped_papers(base_output_folder, [], paper_specific_skipped_pages) 
    
    if already_existed_count > 0:
        print(f"Skipped {already_existed_count} papers that already had output files.")

    return all_successfully_processed_outputs, all_skipped_pages_accumulator

def main(args):
    """
    Evaluate the OCR model on images from specified groups.
    
    Args:
        args: Command line arguments
    """
    tokenizer, model = load_model_and_tokenizer(args.model_name)
    
    papers_dict, skipped_papers_on_load = get_paper_images(args.base_input, args.groups, args.max_images)
    
    if skipped_papers_on_load:
        log_skipped_papers(args.base_output, skipped_papers_on_load)
    
    if not papers_dict:
        print("No valid papers found in the specified groups. Exiting.")
        return
    
    processed_output_files, overall_skipped_pages = process_papers_interleaved(
        model,
        tokenizer,
        papers_dict,
        args.base_output,
        ocr_type=args.type,
        box=args.box,
        color=args.color,
        render=args.render,
        timeout_seconds=args.timeout
    )
    
    print(f"\n--- OCR Process Complete ---")
    print(f"Successfully processed/found existing output for {len(processed_output_files)} papers.")
    if overall_skipped_pages:
        print(f"Encountered {len(overall_skipped_pages)} skipped pages during processing. See log for details.")
    else:
        print("No pages were skipped during processing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="ucaslcl/GOT-OCR2_0")
    parser.add_argument("--base-input", type=str, default="/data/images", help="Base input directory containing group folders")
    parser.add_argument("--base-output", type=str, default="/data/outputs", help="Base output directory for results")
    parser.add_argument("--groups", nargs='+', required=True, help="List of group names to process")
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--box", type=str, default= '')
    parser.add_argument("--color", type=str, default= '')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--max-images", type=int, default=50, help="Maximum number of images per paper")
    parser.add_argument("--timeout", type=int, default=15, help="Maximum time in seconds for generation before timing out")
    args = parser.parse_args()

    os.makedirs(args.base_output, exist_ok=True)

    main(args)
