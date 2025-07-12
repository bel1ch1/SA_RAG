import fitz  # PyMuPDF
from PIL import Image
import io

# Открываем PDF
doc = fitz.open("data/Sinamics_S120_Силовые_части_книжного_формата.pdf")
page = doc.load_page(39)  # Первая страница

# Извлекаем первое изображение
img_list = page.get_images(full=True)
if img_list:
    xref = img_list[0][0]  # XREF первого изображения
    base_image = doc.extract_image(xref)
    img_bytes = base_image["image"]

    # Создаём объект изображения и показываем
    img = Image.open(io.BytesIO(img_bytes))
    img.show()  # Открывает в стандартном просмотрщике ОС
else:
    print("Изображения не найдены.")
