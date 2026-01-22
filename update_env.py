import os
import sys
import re
from github import Github

# 获取环境变量
GITHUB_TOKEN = os.environ.get("MY_GITHUB_TOKEN")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY")
FILE_PATH = ".env"
TARGET_VAR = "STOCK_LIST"

def format_ashare_code(code):
    # 格式化 A 股代码
    code = str(code).strip()
    if "." in code: 
        return code.upper()
    if code.startswith("6"): 
        return f"{code}.SS"
    elif code.startswith("0") or code.startswith("3"): 
        return f"{code}.SZ"
    return code

def update_file():
    # 1. 检查参数
    if len(sys.argv) < 2:
        print("Error: No stock code provided")
        return
    
    # 清洗参数，防止隐形字符
    raw_code = sys.argv[1]
    new_code = format_ashare_code(raw_code.replace('\xa0', '').strip())
    
    print(f"Updating .env to: {new_code}")

    # 2. 检查 Token
    if not GITHUB_TOKEN:
        print("Error: MY_GITHUB_TOKEN is missing")
        return

    # 3. 连接 GitHub
    g = Github(GITHUB_TOKEN)
    try:
        repo = g.get_repo(REPO_NAME)
        contents = repo.get_contents(FILE_PATH)
        decoded_content = contents.decoded_content.decode("utf-8")
    except Exception as e:
        print(f"Error reading .env: {e}")
        return

    # 4. 正则替换内容
    new_content = re.sub(
        fr"^{TARGET_VAR}=.*", 
        f"{TARGET_VAR}={new_code}", 
        decoded_content, 
        flags=re.MULTILINE
    )

    if new_content == decoded_content:
        print("No changes needed.")
        return

    # 5. 提交修改
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
