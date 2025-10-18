import os
import sys
from config_params.read_json import Config, execute_mode

def run_hydrogel_example(json_path):
    print(f"\n--- 하이드로젤 생성 예시 실행 중 ({os.path.basename(json_path)}) ---")

    # maker.json 로드 및 모드 설정
    Config.load_config(json_path)

    # execute_mode 함수 실행
    execute_mode()

    print("\n--- 하이드로젤 생성 예시 완료 ---")

if __name__ == "__main__":
    # 기본 maker.json 파일 또는 커맨드 라인 인자로 받은 파일을 사용
    #default_json_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'maker.json'))
    # 만약 커맨드 라인에 다른 json 파일이 주어지면 그것을 사용
    json_to_run = sys.argv[1]
    run_hydrogel_example(json_to_run)