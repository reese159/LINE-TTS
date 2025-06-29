import pymupdf

def read_pdf(pdf_path):
    """
    Reads a PDF file and returns its text content.
    """
    text = ""
    try:
        pdf_document = pymupdf.open(pdf_path) # Open the PDF file
        # loop through text, print below for testing
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    except Exception as e:
        print(f"An error occurred: {e}")
    return text

