# 该脚本用于搜索项目中所有包含 '3d' 或 '3D' 的地方
# 以判断是否存在强制转换逻辑

import os

def check_string_references(target_dir="."):
    # 查找所有包含 3d 的文件，不区分大小写，但要记录匹配情况
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            # 过滤不需要扫描的文件类型
            if file.endswith(('.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '3d' in content.lower():
                            print(f"在 {file_path} 中找到引用，请检查是否存在大小写冲突")
                except Exception as e:
                    continue

if __name__ == "__main__":
    check_string_references()