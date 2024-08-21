import os
import re
import unicodedata

from natsort import natsorted

def get_list(path: str) -> list:
    """경로에 있는 파일 목록을 반환합니다.

    Args:
        path (str): 파일 목록을 가져올 경로.

    Returns:
        list: 경로 내의 파일 이름이 자연스럽게 정렬된 리스트.
    """
    file_list = [file for file in os.listdir(path) if file != '.DS_Store']
    file_list = normalize_korean_string_list(file_list)
    return natsorted(file_list)

def normalize_korean_string(s: str) -> str:
    """한글 문자열을 통합 정규화(NFC)합니다.

    Args:
        s (str): 정규화할 문자열.

    Returns:
        str: 정규화된 문자열.
    """
    return unicodedata.normalize('NFC', s)

def normalize_korean_string_list(group: list) -> list:
    """한글 문자열 리스트를 통합 정규화(NFC)합니다.

    Args:
        group (list): 정규화할 문자열 리스트.

    Returns:
        list: 정규화된 문자열 리스트.
    """
    return [normalize_korean_string(file) for file in group]

def select_person(name_list: list, specials: list = None, excepts: list = None) -> list:
    """특정 인물을 선택하거나 배제하는 함수입니다.

    Args:
        name_list (list): 전체 인물 리스트.
        specials (Optional[List[str]]): 선택할 특정 인물 리스트.
        excepts (Optional[List[str]]): 배제할 특정 인물 리스트.

    Returns:
        list: 선택된 인물 리스트.
    """
    specials = specials if specials is not None else []
    excepts = excepts if excepts is not None else []

    # specials와 excepts 리스트의 각 원소들을 정규화
    specials = normalize_korean_string_list(specials)
    excepts = normalize_korean_string_list(excepts)

    # special과 excepts 리스트의 각 원소들이 '이름-광역_기초' 형식인지 확인
    for name in specials + excepts:
        if not re.match(r'.*-[가-힣]+_[가-힣]+', name):
            raise ValueError("이름 형식 오류: specials와 excepts 리스트의 각 원소는 '이름-광역_기초' 형식이어야 합니다.")

    if set(specials) & set(excepts):
        raise ValueError("리스트 중복 오류: specials와 excepts에 동일한 이름을 입력할 수 없습니다.")

    # 배제할 이름을 제외한 리스트 생성
    if excepts:
        name_list = [name for name in name_list if name not in excepts]

    # 특정 인물을 선택한 리스트 생성
    if specials:
        name_list = [name for name in name_list if name in specials]

    return name_list