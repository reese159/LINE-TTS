import fitz  

def read_pdf(pdf_path):
    """
    Reads a PDF file and returns its text content.
    """
    text = ""
    
    try:
        pdf_document = fitz.open(stream=pdf_path, filetype="pdf") # Open the PDF file
        # loop through text, print below for testing
        for page_num in range(pdf_document.page_count):
            text += cropped_text(pdf_document, page_num) # Read each page and crop text
        
        pdf_document.close() # Close the document after reading
    except Exception as e:
        print(f"An error occurred: {e}")
    return text.strip()


def cropped_text(pdf_document, page_num):
    """
    Crops header and footer from current page in PDF, 
    returns text after reading
    """
    page = pdf_document.load_page(page_num)
    
    # Set header and footer percentages
    header_height_percent = 0.05
    footer_height_percent = 0.05
    
    # Find new page dimensions
    page_height = page.rect.height
    header_cutoff = page_height * header_height_percent
    footer_cutoff = page_height * (1 - footer_height_percent)
    page_shape = page.rect
    
    # Set new page shape to exclude header and footer
    cropped_page = fitz.Rect(page_shape.x0, header_cutoff, page_shape.x1, footer_cutoff)
    page.set_cropbox(cropped_page)
    
    return page.get_text()