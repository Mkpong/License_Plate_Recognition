import os

folder_path = r"C:\Users\leeja\Desktop\CNNModel\data\image\test\word\22"  # 폴더 경로 설정

# 폴더 내 파일 목록 읽어오기
file_list = os.listdir(folder_path)

# 파일들 정렬
# file_list.sort(key=lambda x: int(x[3:-4]))

# 파일들의 절대 경로 생성
absolute_file_paths = [os.path.join(folder_path, file_name) for file_name in file_list]
output_file = "C:\\Users\\leeja\\Desktop\\CNNModel\\data\\image\\image.txt"
output_file2 = "C:\\Users\\leeja\\Desktop\\CNNModel\\data\\image\\label.txt"

print(absolute_file_paths[0])

# 파일들의 절대 경로 출력
with open(output_file, "a") as file:
    for file_path in absolute_file_paths:
        file.write(file_path+"\n")

with open(output_file2, "a") as file2:
    for file_path in absolute_file_paths:
        file2.write("22\n")