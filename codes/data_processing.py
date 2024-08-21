import os
import re
import glob
import pickle

import pandas as pd
from tqdm import tqdm
from natsort import natsorted

from codes.utils import get_list, select_person, normalize_korean_string

class DataProcessor:
    """데이터를 정제하고 DataFrame으로 재구성하는 클래스입니다."""

    def __init__(self, doc_dir: str = 'data/documents', df_dir: str = 'data/excels',
                 city_path: str = 'data/city.csv', specials: list = None, excepts: list = None):
        """DataProcessor 클래스의 초기화 메서드.

        Args:
            doc_dir (str): 문서 파일이 저장된 디렉토리 경로. 기본값은 'data/documents'입니다.
            df_dir (str): 재구성된 DataFrame이 저장될 디렉토리 경로. 기본값은 'data/excels'입니다.
            city_path (str): 시군구 이름 파일 경로. 기본값은 'data/city.csv'입니다.
        """
        self.doc_dir = doc_dir
        self.df_dir = df_dir
        self.city_path = city_path
        self.specials = specials if specials is not None else []
        self.excepts = excepts if excepts is not None else []

    def reassemble_dataframe(self) -> None:
        """데이터를 정제해 DataFrame으로 재구성하고 Excel 파일로 저장합니다."""
        name_list = get_list(self.doc_dir)
        filtered_list = select_person(name_list, self.specials, self.excepts)

        for name in tqdm(filtered_list, desc="데이터 정제 중"):
            dataset = []
            last_account_name = ""
            file_list = glob.glob(os.path.join(self.doc_dir, name, '*.pickle'))

            for file_path in natsorted(file_list):
                try:
                    with open(file_path, 'rb') as f:
                        document = pickle.load(f)
                    page = document.pages[0]
                    texts = document.text
                    header = self.page_header(page, texts)

                    if any(keyword in header for keyword in ["Page", "page", "페이지"]):
                        account = self.account_name(page, texts) or last_account_name
                        last_account_name = account

                        page_data = self.extract_page_data(page, texts)
                        page_data = self.process_page_data(page_data, texts)
                        page_data = [[account] + row_data for row_data in page_data]
                        page_data = self.normalize_row_data_length(page_data)
                        dataset.extend(page_data)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

            self.save_to_excel(dataset, name)

    def page_header(self, page: object, texts: str) -> str:
        """표가 들어있는 페이지를 구분하기 위해, 페이지 헤더를 추출합니다.

        Args:
            page: 페이지 객체.
            texts (str): 페이지에서 추출한 전체 텍스트.

        Returns:
            str: 페이지 헤더 텍스트.
        """
        return "".join(texts[paragraph.layout.text_anchor.text_segments[0].start_index:
                             paragraph.layout.text_anchor.text_segments[0].end_index]
                       for paragraph in page.paragraphs
                       if paragraph.layout.text_anchor.text_segments)

    def account_name(self, page: object, texts: str) -> str:
        """계정명을 추출합니다.

        Args:
            page: 페이지 객체.
            texts (str): 페이지에서 추출한 전체 텍스트.

        Returns:
            str: 계정명 텍스트.
        """
        forms = []
        for i in range(3):
            try:
                form_text = self.layout_to_text(page.form_fields[i].field_value, texts).strip()
            except (IndexError, AttributeError):
                form_text = ''
            forms.append(form_text)

        account_name = ''.join(forms)

        if any(keyword in account_name for keyword in ["후보", "자산"]):
            return "후보자등 자산"
        elif any(keyword in account_name for keyword in ["후원", "기부"]):
            return "후원회기부금"
        elif "지원" in account_name:
            return "보조금외 지원금"
        elif "보조" in account_name:
            return "보조금"
        else:
            return ""

    def layout_to_text(self, layout: object, texts: str) -> str:
        """레이아웃에서 텍스트를 추출합니다.

        Args:
            layout: 레이아웃 객체.
            texts (str): 페이지에서 추출한 전체 텍스트.

        Returns:
            str: 레이아웃에서 추출된 텍스트.
        """
        return "".join(
            texts[int(segment.start_index):int(segment.end_index)] for segment in layout.text_anchor.text_segments)

    def extract_page_data(self, page: object, texts: str) -> list:
        """페이지 객체에서 데이터를 추출합니다.

        Args:
            page: 페이지 객체.
            texts (str): 페이지에서 추출한 전체 텍스트.

        Returns:
            list: 추출된 페이지 데이터.
        """
        page_data = []
        for row in page.tables[0].body_rows:
            row_data = [re.sub(r'[|\n]', '', self.layout_to_text(cell.layout, texts)).strip() for cell in row.cells]
            row_data = self.remove_header_footer(row_data)
            if len(row_data) > 2:
                page_data.append(row_data)
        return page_data

    def remove_header_footer(self, row_data: list) -> list:
        """칼럼 제목 행과 합계 행을 제거합니다.

        Args:
            row_data (list): 원본 행 데이터.

        Returns:
            list: 불필요한 행이 제거된 데이터.
        """
        # 행에서 한글만 추출한 뒤, 키워드가 2개 이상 포함되어 있으면 제목 행으로 간주
        kor_text = ''.join(re.sub('[^\uAC00-\uD7A3]', '', text) for text in row_data)
        keywords = ['연월일', '내역', '금회', '누계', '잔액', '성명', '생년월일', '주소', '직업', '업종', '전화번호', '영수증', '일련번호']
        if sum(1 for keyword in keywords if keyword in kor_text) >= 2:
            return []

        # 합계 행의 단어를 빼고 한글이 3개 미만이면 합계 행으로 간주
        check_set = set(['합', '계', '영', '수', '증', '생', '략', '분', '첨', '부', '건'])
        remain = set(kor_text) - check_set
        return [] if len(remain) < 3 else row_data

    def process_page_data(self, page_data: list, texts: str) -> list:
        """정규식을 이용해 페이지 데이터를 정제합니다.

        Args:
            page_data (list): 페이지 데이터.
            texts (str): 페이지에서 추출한 전체 텍스트.

        Returns:
            list: 정제된 페이지 데이터.
        """
        date_pattern = r'(?:\d{4}-\d{2}-\d{2})|(?:\d{4}\.\d{2}\.\d{2})|(?:\d{4}/\d{2}/\d{2})'
        item_pattern = r'[가-힣a-zA-Z\[(][가-힣a-zA-Z()\[\] ]*'
        number_pattern = r'^-?\d{1,3}(,\d{3})*'
        large_number_pattern = r'\d{1,3}(,\d{3})+'
        id_number_pattern = r'\d+-\d+(-\d+)?'

        page_data = self.split_date_and_others(page_data, date_pattern, item_pattern, large_number_pattern)
        page_data = self.add_lost_data(page_data, texts, date_pattern, item_pattern)
        page_data = self.validate_front_element(page_data, date_pattern, item_pattern)
        page_data = self.insert_space_based_on_patterns(page_data, number_pattern, id_number_pattern)
        page_data = self.split_number_and_others(page_data, number_pattern, item_pattern)
        page_data = self.split_name_and_id(page_data, item_pattern, id_number_pattern)
        page_data = self.add_space_after_city(page_data, item_pattern)

        return page_data

    def split_date_and_others(self, page_data: list, date_pattern: str, item_pattern: str,
                              large_number_pattern: str) -> list:
        """첫 번째 원소(연월일)에 날짜, 내역, 숫자가 붙어 있는 경우 분리합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            date_pattern (str): 날짜 패턴 정규식.
            item_pattern (str): 내역 패턴 정규식.
            large_number_pattern (str): 숫자 패턴 정규식.

        Returns:
            list: 수정된 페이지 데이터.
        """
        for row_data in page_data:
            first_element = row_data[0]
            second_element = row_data[1]

            date_match = re.search(date_pattern, first_element)
            item_match = re.search(item_pattern, first_element)
            number_match = re.search(large_number_pattern, first_element)

            if date_match and item_match:
                row_data[0] = date_match.group()
                if re.search(item_pattern, second_element):
                    row_data[1] = item_match.group() + second_element
                else:
                    row_data.insert(1, item_match.group())

                if number_match:
                    row_data.insert(2, number_match.group())

        return page_data

    def add_lost_data(self, page_data: list, texts: str, date_pattern: str, item_pattern: str) -> list:
        """페이지 데이터에 날짜와 내역 정보가 없는 경우, 페이지 전체 텍스트에서 찾아서 추가합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            texts (str): 페이지에서 추출한 전체 텍스트.
            date_pattern (str): 날짜 패턴 정규식.
            item_pattern (str): 내역 패턴 정규식.

        Returns:
            list: 날짜와 내역이 추가된 페이지 데이터.
        """
        matches = re.findall(fr'({date_pattern})(.*)', texts)

        if len(matches) == len(page_data):
            for i, row_data in enumerate(page_data):
                date_text, item_text = matches[i]
                if not re.search(date_pattern, row_data[0]):
                    row_data.insert(0, date_text)

                # 대체 텍스트(item_text)에서 정확한 내역만을 뽑아 '정제된 대체 텍스트'(item_text_clean) 준비
                item_text_clean = re.sub(r'[|\n]', '', item_text).strip()
                item_match = re.search(item_pattern, item_text)
                if item_match:
                    item_text_clean = item_match.group()

                # 원소2의 내역이 대체 텍스트(item_text)에 포함된다면, '정제된 대체 텍스트'(item_text_clean)로 교체
                if re.search(item_pattern, row_data[1]):
                    if row_data[1].strip() in item_text:
                        row_data[1] = item_text_clean
                else:
                    row_data.insert(1, item_text_clean)

        return page_data

    def validate_front_element(self, page_data: list, date_pattern: str, item_pattern: str) -> list:
        """첫 번째(연월일)와 두 번째(내역) 원소를 검증하고 정제합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            date_pattern (str): 날짜 패턴 정규식.
            item_pattern (str): 내역 패턴 정규식.

        Returns:
            list: 정제된 페이지 데이터.
        """
        cleaned_data = []
        for row_data in page_data:
            if row_data[0].strip() == '' and row_data[1].strip() == '':
                row_data = row_data[2:]
            elif row_data[0].strip() == '':
                row_data = row_data[1:]

            # 원소1에 날짜 형식이 있으면 날짜만 남기고 나머지 문자 제거
            date_match = re.search(date_pattern, row_data[0])
            if date_match:
                row_data[0] = re.sub(r'[./]', '-', date_match.group())
            # 원소1에 날짜 형식이 없고, 원소2에 날짜 형식이 있으면 원소1 제거
            elif re.search(date_pattern, row_data[1]):
                row_data = row_data[1:]
            # 원소1,2에 날짜 형식이 없으면 원소1에 공백 추가
            else:
                row_data.insert(0, '')

            # 원소2에 내역이 없고 원소3에 내역이 있으면 원소2 제거
            if not re.search(item_pattern, row_data[1]):
                if re.search(item_pattern, row_data[2]):
                    row_data.pop(1)
                else:
                    row_data.insert(1, '')

            cleaned_data.append(row_data)

        return cleaned_data

    def insert_space_based_on_patterns(self, page_data: list, number_pattern: str, id_number_pattern: str) -> list:
        """숫자 칼럼들이 앞으로 당겨져 있다면, 수입 금회에 공백을 삽입합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            number_pattern (str): 숫자 패턴 정규식.
            id_number_pattern (str): 사업자번호 패턴 정규식.

        Returns:
            list: 공백이 삽입된 페이지 데이터.
        """
        for row_data in page_data:
            # index 2~5(수입 금회 ~ 지출 누계)가 모두 number_pattern을 포함하는지 확인
            all_numbers = all(re.search(number_pattern, row_data[i]) for i in range(2, 6))

            # index 6(잔액)에 number_pattern이 없거나, id_number_pattern이 있는지 확인
            number_missing = not re.search(number_pattern, row_data[6]) or re.search(id_number_pattern, row_data[6])

            # 조건이 모두 만족되면 index 2에 공백 삽입
            if all_numbers and number_missing:
                row_data.insert(2, '')
        return page_data

    def split_number_and_others(self, page_data: list, number_pattern: str, item_pattern: str) -> list:
        """일곱 번째 원소(잔액)에 숫자와 성명이 있는 경우 분리합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            number_pattern (str): 숫자 패턴 정규식.
            item_pattern (str): 내역 패턴 정규식.

        Returns:
            list: 분리된 페이지 데이터.
        """
        for row_data in page_data:
            if len(row_data) >= 7:
                seventh_element = row_data[6]
                number_match = re.search(number_pattern, seventh_element)
                item_match = re.search(item_pattern, seventh_element)

                if number_match and item_match:
                    row_data[6] = number_match.group()

                    if len(row_data) >= 8:
                        if re.search(item_pattern, row_data[7]):
                            row_data[7] = item_match.group() + row_data[7]
                        else:
                            row_data.insert(7, item_match.group())
                    else:
                        row_data.append(item_match.group())
        return page_data

    def split_name_and_id(self, page_data: list, item_pattern: str, id_number_pattern: str) -> list:
        """여덟 번째 원소(성명)에 성명과 사업자번호가 붙어 있는 경우 분리합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            item_pattern (str): 내역 패턴 정규식.
            id_number_pattern (str): 사업자번호 패턴 정규식.

        Returns:
            list: 분리된 페이지 데이터.
        """
        for row_data in page_data:
            if len(row_data) >= 8:
                eighth_element = row_data[7]
                item_match = re.search(item_pattern, eighth_element)
                id_match = re.search(id_number_pattern, eighth_element)

                if item_match and id_match:
                    row_data[7] = eighth_element.replace(id_match.group(), '').strip()
                    row_data.insert(8, id_match.group())
        return page_data

    def add_space_after_city(self, page_data: list, item_pattern: str) -> list:
        """주소 칼럼에서 시군구 문자 뒤에 공백을 추가합니다.

        Args:
            page_data (list): 원본 페이지 데이터.
            item_pattern (str): 내역 패턴 정규식.

        Returns:
            list: 공백이 추가된 페이지 데이터.
        """
        cities = pd.read_csv(self.city_path, encoding='cp949')['시군구'].tolist()
        cities_str = '|'.join(re.escape(city) for city in cities)

        for row_data in page_data:
            for i, element in enumerate(row_data):
                if i in {8, 9, 10} and re.search(item_pattern, element):
                    row_data[i] = re.sub(fr'({cities_str})(?!\s)', r'\1 ', element)

        return page_data

    def normalize_row_data_length(self, page_data: list) -> list:
        """행의 길이를 정규화합니다. 14개보다 많으면 자르고, 적으면 공백으로 채웁니다.

        Args:
            page_data (list): 원본 페이지 데이터.

        Returns:
            list: 정규화된 페이지 데이터.
        """
        for i, row_data in enumerate(page_data):
            if len(row_data) > 14:
                page_data[i] = row_data[:14]
            elif len(row_data) < 14:
                page_data[i].extend([''] * (14 - len(row_data)))

        return page_data

    def save_to_excel(self, dataset: list, name: str) -> None:
        """데이터셋을 Dataframe으로 변환한 뒤 Excel 파일로 저장합니다.

        Args:
            dataset (list): 저장할 데이터셋.
            name (str): 국회의원 이름.
        """
        column_name = ['계정명', '연월일', '내역', '수입 금회', '수입 누계', '지출 금회', '지출 누계', '잔액', '성명-법인단체명',
                       '생년월일-사업자번호', '주소-사무소소재지', '직업-업종', '전화번호', '영수증 일련번호']
        df = pd.DataFrame(dataset, columns=column_name)
        df = df.fillna('')
        df = self.clean_name_column(df, '성명-법인단체명', name)
        df = self.clean_financial_columns(df)
        df.to_excel(os.path.join(self.df_dir, f'{name}.xlsx'), index=False)

    def clean_name_column(self, df: pd.DataFrame, column_name: str, name: str) -> pd.DataFrame:
        """특정 칼럼에서 띄어쓰기를 제거하고, 오타를 정정합니다.

        Args:
            df (pd.DataFrame): 원본 DataFrame.
            column_name (str): 수정할 칼럼 이름.
            name (str): 수정 기준이 되는 이름.

        Returns:
            pd.DataFrame: 수정된 DataFrame.
        """
        df[column_name] = df[column_name].str.replace(' ', '')
        real_name = normalize_korean_string(name.split('-')[0])
        df[column_name] = df[column_name].apply(
            lambda x: normalize_korean_string(x) if x is not None else x)
        df[column_name] = df[column_name].str.replace(f'국회의원{real_name[1:]}후원회', f'국회의원{real_name}후원회')
        return df

    def clean_financial_columns(self, df: pd.DataFrame, start_col: int = 3, end_col: int = 8) -> pd.DataFrame:
        """수입 금회 ~ 잔액 칼럼에서 비정상적인 숫자 패턴을 체크하고 수정합니다.

        Args:
            df (pd.DataFrame): 원본 DataFrame.
            start_col (int): 시작 칼럼 인덱스. 기본값은 3.
            end_col (int): 끝 칼럼 인덱스. 기본값은 8.

        Returns:
            pd.DataFrame: 수정된 DataFrame.
        """
        # 수입 금회부터 잔액까지 모두 비어있으면 그 행을 삭제
        df = df[~df.iloc[:, start_col:end_col].eq('').all(axis=1)].copy()

        # 비정상적인 숫자 패턴을 보일 경우 체크 추가
        number_pattern = r'^-?\d{1,3}(,\d{3})*(\.\d+)?$'
        df.iloc[:, start_col:end_col] = df.iloc[:, start_col:end_col].astype(str).apply(
            lambda col: col.map(lambda x: x if x.strip() == '' or re.match(number_pattern, x) else x + ' check')
        )
        return df