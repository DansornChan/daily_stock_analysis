import os
import sys
import re
from github import Github

# 获取环境变量
GITHUB_TOKEN = os.environ.get("MY_GITHUB_TOKEN")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY")
FILE_PATH = ".env"
TARGET_VAR = "STOCK_LIST"

def update_file():
    # 1. 检查参数
    if len(sys.argv) < 2:
        print("Error: No stock code provided")
        return
    
    # 获取原始代码，去掉可能的隐形字符，不做任何后缀修改
    # Akshare/Efinance 通常只需要 6 位数字
    new_code = sys.argv[1].strip().replace('\xa0', '')
    
    print(f"Updating .env to: {new_code}")

    # 2. 连接 GitHub
    if not GITHUB_TOKEN:
        print("Error: MY_GITHUB_TOKEN is missing")
        return

    g = Github(GITHUB_TOKEN)
    try:
        repo = g.get_repo(REPO_NAME)
        contents = repo.get_contents(FILE_PATH)
        decoded_content = contents.decoded_content.decode("utf-8")
    except Exception as e:
        print(f"Error reading .env: {e}")
        return

    # 3. 正则替换
    new_content = re.sub(
        fr"^{TARGET_VAR}=.*", 
        f"{TARGET_VAR}={new_code}", 
        decoded_content, 
        flags=re.MULTILINE
    )

    if new_content == decoded_content:
        print("No changes needed.")
        return

    # 4. 提交
    repo.update_file(
        path=contents.path,
        message=f"Bot Update: {new_code}",
        content=new_content,
        sha=contents.sha,
        branch="main"
    )
    print("Update successful!")

if __name__ == "__main__":
    update_file()
