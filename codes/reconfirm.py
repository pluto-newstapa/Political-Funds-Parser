import os
import glob

import pandas as pd
from codes.utils import get_list

class FinancialDataValidator:
    def __init__(self, df_dir: str = 'data/excels'):
        """FinancialDataValidator 초기화 메서드.

        Args:
            df_dir (str): 엑셀 파일이 저장된 디렉토리 경로. 기본값은 'data/excels'.
        """
        self.df_dir = df_dir

    def validate_financial_data(self, amount) -> None:
        """엑셀 파일에서 금액 칼럼이 내적으로 완결성을 가지는지 검증합니다.

        Args:
            amount (str): 검증할 금액 칼럼. '수입', '지출', '잔액' 중 하나.

        """
        file_list = [file for file in get_list(self.df_dir) if file.endswith('.xlsx')]
        for file in file_list:
            file_path = os.path.join(self.df_dir, file)
            try:
                df = pd.read_excel(file_path, engine='openpyxl', dtype=str)

                # 숫자로 변환할 수 있는 열을 숫자로 변환
                df = self._convert_to_number(df, file)

                # 계정명으로 그룹화하고, 각 계정별 첫 번째 행의 인덱스를 가져옴
                df_grouped = df.groupby('계정명')
                first_index = df_grouped.head(1).index

                # 금액 칼럼 검증
                self._check_amount(df, file, amount, first_index)

            except Exception as e:
                print(f"{file}: {e}")

    def _convert_to_number(self, df: pd.DataFrame, file: str) -> pd.DataFrame:
        """숫자로 변환할 수 있는 열을 숫자로 변환합니다.

        Args:
            df (pd.DataFrame): 변환할 DataFrame.
            file (str): 변환할 파일 이름.

        Returns:
            pd.DataFrame: 변환된 DataFrame.
        """

        cols = ['수입 금회', '수입 누계', '지출 금회', '지출 누계', '잔액']
        df[cols] = df[cols].fillna('0').replace(',', '', regex=True)

        # 안전하게 숫자로 변환 (오류가 있으면 NaN 반환)
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # 변환 오류 확인 및 로깅
        if df[cols].isnull().any().any():
            problematic_rows = df[df[cols].isnull().any(axis=1)]
            for index, row in problematic_rows.iterrows():
                for col in cols:
                    if pd.isna(row[col]):
                        print(f"{file} {index+2}행 : {row['연월일']} {row['내역']} / {col} 변환 실패 - 숫자로 변환할 수 없는 값")

        return df

    def _check_amount(self, df: pd.DataFrame, file: str, amount: str, first_index: pd.Index) -> None:
        """금액 칼럼끼리 비교해 내적 완결성을 검증합니다. 계정별 첫 번째 행은 예외 처리합니다.
        수입 확인 = 수입 누계 - 수입 누계.shift(1)
        지출 확인 = 지출 누계 - 지출 누계.shift(1)
        잔액 확인 = 수입 누계 - 지출 누계

        Args:
            df (pd.DataFrame): 검증할 DataFrame.
            file (str): 검증할 파일 이름.
            amount (str): 검증할 계정명.
            first_index (pd.Index): 계정별 첫 번째 행의 index.
        """
        if amount in ['수입', '지출']:
            this_amount = f'{amount} 금회'
            sum_amount = f'{amount} 누계'
            check_amount = f'{amount} 확인'

            # 수입/지출 확인 = 수입/지출 누계 - 수입/지출 누계.shift(1)
            df[check_amount] = df[sum_amount] - df[sum_amount].shift(1)

            # 계정별 첫 번째 행은 예외 처리
            df.loc[first_index, check_amount] = df.loc[first_index, this_amount]

            # 정수로 변환
            df[check_amount] = df[check_amount].astype(int)

            # 불일치하는 행 출력
            mismatch = ~df[check_amount].eq(df[this_amount])
            if mismatch.any():
                print(f"{file} : {this_amount} & {check_amount} 불일치")
                mismatch_data = df.loc[mismatch, ['연월일', '내역', this_amount, check_amount]].copy()
                mismatch_data.index = mismatch_data.index + 2
                print(mismatch_data)

        elif amount == '잔액':
            df['잔액 확인'] = df['수입 누계'] - df['지출 누계']
            mismatch = ~df['잔액 확인'].eq(df['잔액'])
            if mismatch.any():
                print(f"{file} : 잔액 & 잔액 확인 불일치")
                mismatch_data = df.loc[mismatch, ['연월일', '내역', '잔액', '잔액 확인']].copy()
                mismatch_data.index = mismatch_data.index + 2
                print(mismatch_data)
        else:
            pass

def main():
    # 인스턴스 생성
    validator = FinancialDataValidator()

    # 수치화, 수입, 지출, 잔액 각각 검증 (순차 실행 권장)
    validator.validate_financial_data('수치화')
    # validator.validate_financial_data('수입')
    # validator.validate_financial_data('지출')
    # validator.validate_financial_data('잔액')

if __name__ == "__main__":
    main()


