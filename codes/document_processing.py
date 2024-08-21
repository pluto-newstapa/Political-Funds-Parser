import os
import re
import glob
import pickle

from tqdm import tqdm
from natsort import natsorted
from google.cloud import documentai
from google.api_core.client_options import ClientOptions

from codes.utils import get_list, normalize_korean_string_list, select_person

os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

class DocumentProcessor:
    """Google Cloud Document AI를 통해 OCR을 수행하는 클래스입니다."""

    def __init__(self, project_id: str, location: str, processor_id: str, image_dir: str = 'data/images',
                 doc_dir: str = 'data/documents', specials: list = None, excepts: list = None):
        """DocumentProcessor 초기화 메서드.

        Args:
            project_id (str): Google Cloud 프로젝트 ID.
            location (str): Document AI 프로세서의 위치 (예: "us", "eu").
            processor_id (str): Document AI 프로세서 ID.
            image_dir (str): 이미지 파일이 저장된 디렉토리 경로. 기본값은 'data/images'입니다.
            doc_dir (str): 변환된 Document 객체를 저장할 디렉토리 경로. 기본값은 'data/documents'입니다.
            specials (Optional[List[str]]): 선택할 특정 인물 리스트.
            excepts (Optional[List[str]]): 배제할 특정 인물 리스트.
        """
        if not project_id or not location or not processor_id:
            raise ValueError("project_id, location, processor_id 정보가 있어야 합니다.")

        self.client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        )
        self.name = self.client.processor_version_path(project_id, location, processor_id, "stable")
        self.image_dir = image_dir
        self.doc_dir = doc_dir
        self.specials = specials if specials is not None else []
        self.excepts = excepts if excepts is not None else []

    def run_documentai_ocr(self) -> None:
        """이미지 파일에서 텍스트를 추출하여 Document 객체로 변환합니다."""
        name_list = get_list(self.image_dir)
        filtered_list = select_person(name_list, self.specials, self.excepts)

        for name in tqdm(filtered_list, desc="Document AI 실행 중"):
            file_list = glob.glob(os.path.join(self.image_dir, name, '*.png'))

            try:
                for img_path in natsorted(file_list):
                    with open(img_path, "rb") as f:
                        image = f.read()

                    # Document AI로 이미지를 분석하여 Document 객체를 반환
                    document = self.get_document(image)

                    # Document 객체를 pickle 파일로 저장
                    doc_path = os.path.join(self.doc_dir, name)
                    page_num = os.path.splitext(os.path.basename(img_path))[0]
                    self.save_pickle(doc_path, page_num, document)
            except Exception as e:
                print(f"Document AI 실행 중 오류 발생: {e}")

    def get_document(self, image: bytes) -> documentai.Document:
        """Document AI로 이미지를 분석하여 Document 객체를 반환합니다.

        Args:
            image (bytes): 이미지 데이터.

        Returns:
            documentai.Document: 분석된 Document 객체.
        """
        raw_document = documentai.RawDocument(content=image, mime_type="image/png")
        request = documentai.ProcessRequest(name=self.name, raw_document=raw_document)
        return self.client.process_document(request=request).document

    def save_pickle(self, doc_path: str, page_num: str, document: documentai.Document) -> None:
        """Document 객체를 pickle 파일로 저장합니다.

        Args:
            doc_path (str): 저장할 경로.
            page_num (str): 페이지 번호.
            document (documentai.Document): 저장할 Document 객체.
        """
        os.makedirs(doc_path, exist_ok=True)
        output_path = os.path.join(doc_path, f'{page_num}.pickle')
        with open(output_path, 'wb') as f:
            pickle.dump(document, f)
