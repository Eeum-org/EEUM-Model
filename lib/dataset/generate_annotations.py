import os
import json
import csv
from lib.config.settings import DATA_DIR

def _find_video_folder(data_dir):
    """'video'를 포함하는 하위 폴더 탐색"""
    video_folders = []
    for subdir in os.listdir(data_dir):
        next_path = os.path.join(data_dir, subdir)
        if os.path.exists(next_path) and os.path.isdir(next_path):
            for item in os.listdir(next_path):
                path = os.path.join(next_path, item)
                if os.path.isdir(path) and 'video' in item:
                    video_folders.append(path)

    return video_folders if video_folders else None

def generate_annotations_csv(split='train'):
    """
    annotations/[prefix]_annotations.csv를 자동 생성.
    """
    split_dir = os.path.join(DATA_DIR, split)
    
    # 비디오 폴더 확인
    video_folder_names = _find_video_folder(split_dir)
    if not video_folder_names:
        raise FileNotFoundError(f"No video folder found in '{split_dir}'")

    # annotations 폴더 생성
    annotations_dir = os.path.join(split_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)
    
    
    for video_folder in video_folder_names:
        # 비디오 폴더별 csv 파일명 지정
        csv_filename = os.path.split(video_folder)[1].replace('_video', '_annotations.csv')
        output_csv_path = os.path.join(annotations_dir, csv_filename)
        annotations = []

        # morpheme 폴더 탐색
        morpheme_dir = video_folder.replace("_video", "_morpheme")
        if not os.path.exists(morpheme_dir):
            raise FileNotFoundError(f"Morpheme directory not found: '{morpheme_dir}'")
        print(f"Scanning morpheme directory: {morpheme_dir}")

        base_folder = os.path.split(os.path.split(video_folder)[0])[1]
        # morpheme 폴더 내 하위 디렉터리 순회
        for file in sorted(os.listdir(morpheme_dir)):
            # F는 무조건 존재하므로, _F를 기준으로
            if file.endswith('_F_morpheme.json'):
                base_name = file.replace('_F_morpheme.json', '')
                json_path = os.path.join(morpheme_dir, file)
                print(f"Processing file: {file} -> base_name: {base_name}")
                
                try:
                    meanings = []
                    attributes = []
                    start_time = None
                    end_time = None
                    with open(json_path, 'r', encoding='utf-8') as f:
                        loaded_json = json.load(f)
                        # 'data' 키가 있는지 확인
                        if 'data' in loaded_json:
                            for check_data in loaded_json['data']:
                                if "start" in check_data:
                                    start_time = check_data["start"]
                                if "end" in check_data:
                                    end_time = check_data["end"]
                                if 'attributes' in check_data:
                                    for attribute in check_data['attributes']:
                                        for k, v in attribute.items():
                                            if k == "name":
                                                meanings.append(v)
                                            else:
                                                attributes.extend(v)
                            meaning = " ".join(meanings)
                        else:
                            print(f"Warning: No 'data' key found in {json_path}")
                            meaning = "<unk>"
                            
                except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                    print(f"Error processing {json_path}: {e}")
                    meaning = "<unk>"
                
                if meaning.strip():  # 빈 문자열이 아닌 경우만 추가
                    annotations.append({'file_name': os.path.join(base_folder, base_name), 'meaning': meaning, 'attributes' : attributes if attributes else None,\
                                        "start" : start_time if start_time else None, "end" : end_time if end_time else None})
                    print(f"Added annotation: {base_name} -> {meaning}")

        # CSV 파일 작성
        if not annotations:
            print("Warning: No annotations found! CSV will be empty.")
        
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'meaning', 'attributes', 'start', 'end'])
            writer.writeheader()
            annotations.sort(key = lambda x : x["file_name"])
            writer.writerows(annotations)
        
        print(f"Successfully generated '{output_csv_path}' with {len(annotations)} entries.")

if __name__ == '__main__':
    generate_annotations_csv('train')
    # generate_annotations_csv('test')