import os
from huggingface_hub import snapshot_download


def download_model():
    model_name = "BAAI/bge-m3"
    # 获取目标目录，默认为当前目录下的 embedding-models/bge-m3
    local_dir = os.environ.get("MODEL_SAVE_PATH", "./embedding-models/bge-m3")

    print(f"正在从镜像站下载模型 {model_name} 到 {local_dir}...")

    # 设置镜像站环境变量（脚本内设置作为双重保险）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        # 可以根据需要排除不必要的文件，如 .bin (如果已经有 .safetensors)
        ignore_patterns=["*.DS_Store", "imgs/.DS_Store"]
    )
    print("模型下载完成！")


if __name__ == "__main__":
    download_model()
