import pymupdf # type: ignore

pdf_document = pymupdf.open(r'docs\Project Spec.pdf') # Open the PDF file

# loop through text, print below for testing
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    text = page.get_text()
    print(text)