from pypdf import PdfReader

def extracting_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    full_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)
    return "\n".join(full_text)

def save_to_txt(text: str, output_path: str):
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(text)
        print("Saved")


pdf_path = "data/Sinamics_S120_Силовые_части_книжного_формата.pdf"
output_file = "extracted_data/sinamics_1.txt"

extracted_text = extracting_text_from_pdf(pdf_path)

save_to_txt(extracted_text, output_file)
