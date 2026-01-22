import os
import sys
import re
from github import Github # 依赖 PyGithub

# 从 GitHub Action 的环境变量里获取 Token 和 仓库名
# 稍后我们在设置里配置这些
GITHUB_TOKEN = os.environ.get("MY_GITHUB_TOKEN")
REPO_NAME = os.environ.get("GITHUB_REPOSITORY") 
FILE_PATH = ".env"
TARGET_VAR = "STOCK_LIST"

def format_ashare_code(code):
    code = str(code).strip()
    if "." in code: return code.upper()
    if code.startswith("6"): return f"{code}.SS"
    elif code.startswith("0") or code.startswith("3"): return f"{code}.SZ"
    return code

def update_file():
    # 获取命令行传入的股票代码 (比如从 iOS 传来的)
    if len(sys.argv) < 2:
        print("错误：没有收到股票代码")
        return
    
    raw_code = sys.argv[1]
    new_code = format_ashare_code(raw_code)
    
    print(f"正在将 .env 更新为: {new_code}")

    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)
    contents = repo.get_contents(FILE_PATH)
    decoded_content = contents.decoded_content.decode("utf-8")

    # 正则替换
    new_content = re.sub(
        fr"^{TARGET_VAR}=.*", 
        f"{TARGET_VAR}={new_code}", 
        decoded_content, 
        flags=re.MULTILINE
    )

    if new_content == decoded_content:
        print("内容未变，跳过更新")
        return

    repo.update_file(
        path=contents.path,
        message=f"iOS Update: {new_code}",
        content=new_content,
        sha=contents.sha,
        branch="main"
    )
    print("更新成功！")

if __name__ == "__main__":
    update_file()

