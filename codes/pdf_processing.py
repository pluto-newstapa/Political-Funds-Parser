import os
import re
import glob

from tqdm import tqdm
from natsort import natsorted
from pdf2image import convert_from_path

from codes.utils import get_list, normalize_korean_string, select_person

class FilenameStandardizer:
    """PDF 파일의 이름을 표준화하는 클래스입니다."""

    def __init__(self, pdf_dir: str = 'data/raw_pdfs'):
        """FilenameStandardizer 클래스의 초기화 메서드.

        Args:
            pdf_dir (str): PDF 파일이 저장된 디렉토리 경로. 기본값은 'data/raw_pdfs'입니다.
        """
        self.pdf_dir = pdf_dir

    def standardize_filename(self) -> None:
        """PDF 파일을 모두 불러와 이름을 표준화합니다."""
        pdf_files = glob.glob(os.path.join(self.pdf_dir, '**', '*.pdf'), recursive=True)
        error_files = []

        for file_path in pdf_files:
            folder, file = os.path.split(file_path)

            norm_file = normalize_korean_string(file).replace(' ', '')
            standardized_file = self.match_and_standardize(norm_file)

            if standardized_file:
                new_path = os.path.join(folder, standardized_file)
                os.rename(file_path, new_path)
                if standardized_file != norm_file:
                    print(f"파일 이름을 수정했습니다: {file} -> {standardized_file}")
            else:
                error_files.append(file)

        self.handle_errors(error_files)

    def match_and_standardize(self, norm_file: str) -> str:
        """
        파일 이름이 특정 패턴과 일치하는지 확인하고 표준화합니다.

        Args:
            norm_file (str): 표준화할 파일 이름.

        Returns:
            str: 표준화된 파일 이름. 표준화할 수 없으면 None을 반환.
        """
        patterns = [
            (r'(.*?)국회의원([가-힣]+-비례)(\(.*)', lambda m: m.group(0)),
            (r'(.*?)국회의원([가-힣]+-[가-힣]+_[가-힣]+)(\(.*)', lambda m: m.group(0)),
            (r'(.*?)국회의원([가-힣]+-[가-힣]+)-([가-힣]+)(\(.*)',
             lambda m: f"{m.group(1)}국회의원{m.group(2)}_{m.group(3)}{m.group(4)}"),
            (r'(.*?)국회의원([가-힣]+)_([가-힣]+)_([가-힣]+)(\(.*)',
             lambda m: f"{m.group(1)}국회의원{m.group(2)}-{m.group(3)}_{m.group(4)}{m.group(5)}")
        ]

        for pattern, formatter in patterns:
            match = re.match(pattern, norm_file)
            if match:
                return formatter(match)

        return None

    def handle_errors(self, error_files: list) -> None:
        """
        파일 이름 표준화 실패 시 오류 메시지를 출력합니다.

        Args:
            error_files (list): 표준화 실패한 파일들의 목록.
        """
        if error_files:
            error_message = (
                "아래 파일은 표준화할 수 없습니다. 파일 이름을 직접 수정한 뒤 다시 실행해주세요.\n"
                "지역구 의원은 '국회의원이름-광역_기초(00매)', 비례 의원은 '국회의원이름-비례(00매)' 형식을 포함해야 합니다.\n"
            )
            e = '\n'.join(error_files)
            raise ValueError(f"{error_message}\n{e}")
        else:
            print("표준화 성공: 모든 파일의 이름을 표준화했습니다. 다음 단계로 진행하세요.")


class PDFConverter:
    """PDF 파일을 이미지로 변환하는 클래스입니다."""

    def __init__(self, pdf_dir: str = 'data/raw_pdfs', image_dir: str = 'data/images',
                 dpi: int = 300, specials: list = None, excepts: list = None):
        """PDFConverter 클래스의 초기화 메서드.

        Args:
            pdf_dir (str): PDF 파일이 저장된 디렉토리 경로. 기본값은 'data/raw_pdfs'입니다.
            image_dir (str): 이미지가 저장될 디렉토리 경로. 기본값은 'data/images'입니다.
            dpi (int): 이미지 변환 시 적용될 DPI 값. 기본값은 300입니다.
            specials (Optional[List[str]]): 선택할 특정 인물 리스트.
            excepts (Optional[List[str]]): 배제할 특정 인물 리스트.
        """
        self.pdf_dir = pdf_dir
        self.image_dir = image_dir
        self.dpi = dpi
        self.specials = specials if specials is not None else []
        self.excepts = excepts if excepts is not None else []

    def convert_pdf_to_images(self) -> None:
        """PDF 파일을 이미지로 변환합니다."""

        name_dict = self.name_with_path()
        name_list = list(name_dict.keys())
        filtered_list = select_person(name_list, self.specials, self.excepts)
        filtered_dict = {k: v for k, v in name_dict.items() if k in filtered_list}
        self.split_and_convert(filtered_dict)

    def name_with_path(self) -> dict:
        """PDF 파일 이름과 경로를 딕셔너리로 반환합니다.

        Returns:
            dict: PDF 파일 이름을 키로, 경로를 값으로 갖는 딕셔너리.
        """
        name_dict = {}
        pdf_files = glob.glob(os.path.join(self.pdf_dir, '**', '*.pdf'), recursive=True)

        for file_path in natsorted(pdf_files):
            file = os.path.basename(file_path)
            try:
                name = file.split('국회의원')[1].split('(')[0]
                name_dict[name] = file_path
            except Exception as e:
                raise ValueError(f"아래 파일 이름에 오류가 있습니다. 파일 이름 표준화를 다시 진행하세요. ({e})\n{file_path}")

        return name_dict

    def split_and_convert(self, name_dict: dict) -> None:
        """PDF 파일을 그레이스케일 이미지로 변환합니다.

        Args:
            name_dict (dict): PDF 파일 이름을 키로, 경로를 값으로 갖는 딕셔너리.
        """
        for name, path in tqdm(name_dict.items(), desc="PDF를 이미지로 변환 중"):
            image_dir = os.path.join(self.image_dir, name)
            os.makedirs(image_dir, exist_ok=True)
            try:
                images = convert_from_path(path, dpi=self.dpi)
                for i, image in enumerate(images):
                    grayscale_image = image.convert('L')
                    image_path = os.path.join(image_dir, f'page_{i + 1}.png')
                    grayscale_image.save(image_path, 'PNG', quality=100, dpi=(self.dpi, self.dpi))
            except Exception as e:
                print(f"Error processing file {path}: {e}")




