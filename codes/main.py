import os
import sys

from codes.pdf_processing import FilenameStandardizer, PDFConverter
from codes.image_processing import BlurryImageDetector, ImageRotator
from codes.document_processing import DocumentProcessor
from codes.data_processing import DataProcessor


def standardize_filename(pdf_dir='data/raw_pdfs'):
    pdf_processor = FilenameStandardizer(pdf_dir)
    pdf_processor.standardize_filename()

def convert_pdf_to_images(pdf_dir='data/raw_pdfs', image_dir='data/images', dpi=300, specials=None, excepts=None):
    pdf_processor = PDFConverter(pdf_dir, image_dir, dpi, specials, excepts)
    pdf_processor.convert_pdf_to_images()

def improve_blurry_images(pdf_dir='data/raw_pdfs', image_dir='data/images', threshold=50, specials=None, excepts=None):
    image_processor = BlurryImageDetector(pdf_dir, image_dir, threshold, specials, excepts)
    image_processor.improve_blurry_images()

def rotate_twisted_images(image_dir='data/images', specials=None, excepts=None):
    image_rotator = ImageRotator(image_dir, specials, excepts)
    image_rotator.rotate_twisted_images()

def run_documentai_ocr(image_dir='data/images', doc_dir='data/documents', specials=None, excepts=None):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "YOUR_CREDENTIAL_FILE_PATH"  # (ex) "credentials.json"
    project_id = "YOUR_PROJECT_ID"  # Google Cloud project ID
    location = "YOUR_PROCESSOR_LOCATION"  # Format is "us" or "eu"
    processor_id = "YOUR_PROCESSOR_ID"  # Create processor before running sample

    doc_processor = DocumentProcessor(project_id, location, processor_id, image_dir, doc_dir, specials, excepts)
    doc_processor.run_documentai_ocr()

def reassemble_dataframe(doc_dir='data/documents', df_dir='data/excels', city_path='data/city.csv', specials=None, excepts=None):
    data_processor = DataProcessor(doc_dir, df_dir, city_path, specials, excepts)
    data_processor.reassemble_dataframe()


# 실행 안내
if __name__ == "__main__":
    # 아래의 단계를 순서대로 실행하세요. (순차 실행 권장)
    standardize_filename()  # step 1
    # convert_pdf_to_images()  # step 2
    # improve_blurry_images()  # step 3
    # rotate_twisted_images()  # step 4
    # run_documentai_ocr()  # step 5
    # reassemble_dataframe()  # step 6
