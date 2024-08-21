import os
import sys
import math
import glob
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from natsort import natsorted

from codes.pdf_processing import PDFConverter
from codes.utils import get_list, select_person


class BlurryImageDetector:
    """흐릿한 이미지를 감지해 고해상도로 변환하는 클래스입니다."""

    def __init__(self, pdf_dir: str = 'data/raw_pdfs', image_dir: str = 'data/images',
                 threshold: float = 50, specials: list = None, excepts: list = None):
        """BlurryImageDetector 클래스 초기화 메서드.

        Args:
            pdf_dir (str): PDF 파일이 저장된 디렉토리 경로. 기본값은 'data/raw_pdfs'입니다.
            image_dir (str): 이미지 파일이 저장된 디렉토리 경로. 기본값은 'data/images'입니다.
            threshold (float): 흐릿함을 판단할 임계값. 기본값은 50입니다.
            specials (Optional[List[str]]): 선택할 특정 인물 리스트.
            excepts (Optional[List[str]]): 배제할 특정 인물 리스트.
        """
        self.pdf_dir = pdf_dir
        self.image_dir = image_dir
        self.threshold = threshold
        self.specials = specials if specials is not None else []
        self.excepts = excepts if excepts is not None else []

    def improve_blurry_images(self) -> None:
        """흐릿한 이미지를 고해상도로 다시 변환합니다."""
        name_list = get_list(self.image_dir)
        filtered_list = select_person(name_list, self.specials, self.excepts)
        blurry_list = self.detect_blurry_files(filtered_list)

        converter = PDFConverter(pdf_dir=self.pdf_dir, image_dir=self.image_dir, dpi=600)
        name_dict = converter.name_with_path()
        blurry_dict = {k: v for k, v in name_dict.items() if k in blurry_list}
        print("\n흐릿한 파일을 고해상도로 다시 변환합니다.")
        converter.split_and_convert(blurry_dict)

    def detect_blurry_files(self, name_list: list) -> list:
        """흐릿한 이미지가 포함된 파일을 감지합니다.

        Args:
            name_list (list): 파일 이름 리스트.

        Returns:
            list: 흐릿한 이미지 파일을 포함한 폴더 이름 리스트.
        """
        blurry_list = []
        folder_list = get_list(self.image_dir)
        for folder in tqdm(folder_list, desc="흐릿한 파일 감지 중"):
            if folder not in name_list:
                continue

            blurry_count = self._count_blurry_images_in_folder(folder)
            if blurry_count >= 5:
                blurry_list.append(folder)

        return blurry_list

    def _count_blurry_images_in_folder(self, folder: str) -> int:
        """폴더 내의 흐릿한 이미지 개수를 셉니다.

        Args:
            folder (str): 폴더 이름.

        Returns:
            int: 흐릿한 이미지 개수.
        """
        blurry_count = 0
        file_list = glob.glob(os.path.join(self.image_dir, folder, '*.png'))

        for image_path in file_list:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            is_blurry, _ = self.detect_blurry_image(image)

            if is_blurry:
                blurry_count += 1

        return blurry_count

    def detect_blurry_image(self, image: np.ndarray) -> Tuple[bool, float]:
        """이미지가 흐릿한지 감지합니다.

        Args:
            image (np.ndarray): 이미지 데이터.

        Returns:
            tuple: (이미지가 흐릿한지 여부, 라플라시안 값)
        """
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        return laplacian_var < self.threshold, laplacian_var


class ImageRotator:
    """이미지를 회전하여 수평을 맞추는 클래스입니다."""

    def __init__(self, image_dir: str = 'data/images', specials: list = None, excepts: list = None):
        """ImageRotator 클래스 초기화 메서드.

        Args:
            image_dir (str): 이미지 파일이 저장된 디렉토리 경로. 기본값은 'data/images'입니다.
            specials (Optional[List[str]]): 선택할 특정 인물 리스트.
            excepts (Optional[List[str]]): 배제할 특정 인물 리스트.
        """
        self.image_dir = image_dir
        self.specials = specials if specials is not None else []
        self.excepts = excepts if excepts is not None else []

    def rotate_twisted_images(self) -> None:
        """이미지의 비틀림 각도를 측정한 뒤, 그 각도만큼 반대로 회전시켜 수평을 맞춥니다."""
        name_list = get_list(self.image_dir)
        filtered_list = select_person(name_list, self.specials, self.excepts)

        for name in tqdm(filtered_list, desc="이미지의 수평을 맞추는 중"):
            file_list = glob.glob(os.path.join(self.image_dir, name, '*.png'))
            previous_angles = []

            for image_path in natsorted(file_list):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # 최근 이미지 5장의 각도 유지
                if len(previous_angles) >= 5:
                    previous_angles = previous_angles[-5:]

                # 이미지 회전 각도 계산
                angle_degrees = self.get_rotation_angle(image, previous_angles)
                rotated_img = self.rotate_image(image, angle_degrees)

                # 원래 이미지의 DPI 가져오기
                dpi = self.get_image_dpi(image_path)

                # 회전된 이미지 저장
                rotated_img.save(image_path, 'PNG', quality=100, dpi=dpi)
                previous_angles.append(angle_degrees)

    def get_rotation_angle(self, image: np.ndarray, previous_angles: list) -> float:
        """이미지에서 가장 긴 가로선 10개를 추출해 수평과 이루는 각을 구하고, 그 평균값으로 비틀림 각도를 계산합니다.

        Args:
            image (np.ndarray): 이미지 데이터.
            previous_angles (list): 이전 이미지의 비틀림 각도가 담긴 리스트.

        Returns:
            float: 이미지의 회전 각도.
        """
        edges = cv2.Canny(image, threshold1=50, threshold2=150)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
        top_lines = [(0, 0)] * 10

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # 가로선 판단: x 변화량이 y 변화량보다 클 경우
                if abs(x2 - x1) > abs(y2 - y1):
                    angle_radians = math.atan2(y2 - y1, x2 - x1)
                    angle_degrees = math.degrees(angle_radians)
                    # 현재 선을 최대 길이 선 리스트에 추가
                    top_lines.append((line_length, angle_degrees))
                    top_lines.sort(reverse=True)
                    top_lines = top_lines[:10]

        # 가로선 10개가 수평과 이루는 각도의 평균 계산
        angles = [angle for length, angle in top_lines if length > 0]
        if not angles:
            # 유효한 선이 하나도 없을 경우, 이전 각도들의 평균을 사용
            return sum(previous_angles) / len(previous_angles) if previous_angles else 0

        return sum(angles) / len(angles)

    def rotate_image(self, image: np.ndarray, angle: float) -> Image:
        """이미지를 회전시킵니다.

        Args:
            image (np.ndarray): 회전할 이미지 데이터.
            angle (float): 회전할 각도.

        Returns:
            Image: 회전된 이미지.
        """
        pil_image = Image.fromarray(image)
        return pil_image.rotate(angle, expand=True, resample=Image.BICUBIC)

    def get_image_dpi(self, image_path: str) -> Tuple[int, int]:
        """이미지의 DPI를 가져옵니다.

        Args:
            image_path (str): 이미지 파일의 경로.

        Returns:
            tuple: DPI 정보가 없을 경우 기본값은 (300, 300)
        """
        try:
            with Image.open(image_path) as img:
                dpi = img.info.get('dpi', (300, 300))
        except IOError as e:
            print(f"Failed to open image {image_path}: {e}")
            dpi = (300, 300)
        return dpi