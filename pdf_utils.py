import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file, start_page=None, end_page=None):
    """
    Extracts text from PDF between page ranges.
    pdf_file -> file uploaded from Streamlit (BytesIO)
    start_page, end_page are 1-indexed
    """
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    total_pages = doc.page_count

    # Adjust page range
    start = (start_page - 1) if start_page else 0
    end = end_page if end_page else total_pages

    text = ""
    for page_num in range(start, end):
        page = doc.load_page(page_num)
        text += page.get_text()

    return text
