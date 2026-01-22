"""
LangGraph 兼容的工具模块。

本模块使用 Pydantic 定义强类型参数 Schema，并保留核心的截断逻辑以确保安全。
这些工具将被 LangGraph 的 ToolNode 使用，通过 LLM 的 Native Function Calling 调用。
"""

from __future__ import annotations

import os
import shutil
import select
import signal
import subprocess
import time
from typing import Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# ============================================================================
# 截断常量：与原 safe_tools_toon.py 保持一致
# ============================================================================
MAX_LIST_ITEMS = 20
MAX_READFILE_CHARS_DEFAULT = 1000
MAX_READFILE_CHARS_CODE = 20000
MAX_STDOUT_CHARS = 4000
MAX_STDERR_CHARS = 2000
MAX_STDERR_CHARS_FAILURE = 10000


# ============================================================================
# Pydantic Input Schemas (强类型参数定义)
# ============================================================================
class ListDirectoryInput(BaseModel):
    """list_directory 工具的输入参数。"""
    path: str = Field(description="要列举内容的目录路径 (绝对路径)")


class ReadFileInput(BaseModel):
    """read_file 工具的输入参数。"""
    file_path: str = Field(description="要读取的文件路径 (绝对路径)")


class WriteFileInput(BaseModel):
    """write_file 工具的输入参数。"""
    path: str = Field(description="要写入的文件路径 (绝对路径)")
    content: str = Field(description="要写入文件的完整文本内容")


class TerminalInput(BaseModel):
    """terminal 工具的输入参数。"""
    command: str = Field(description="要执行的 shell 命令")


# ============================================================================
# 工具实现
# ============================================================================
class ListDirectoryTool(BaseTool):
    """
    安全版目录列举工具。

    自动截断超长列表，返回清晰的结构化文本。
    """
    name: str = "list_directory"
    description: str = (
        "List the contents of a directory. Returns file and folder names. "
        "Use this to explore the file system structure. "
        "Input: the absolute path to a directory."
    )
    args_schema: Type[BaseModel] = ListDirectoryInput

    def _run(self, path: str) -> str:
        """列举目录内容。"""
        if not os.path.exists(path):
            return f"[ERROR] Path does not exist: {path}"
        if not os.path.isdir(path):
            return f"[ERROR] Path is not a directory: {path}"

        try:
            items = os.listdir(path)
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] Failed to list directory: {e}"

        total = len(items)
        truncated = total > MAX_LIST_ITEMS

        result_lines = [f"Directory: {path}", f"Total items: {total}"]
        
        if truncated:
            result_lines.append("(Summary view for > 20 items)")
            
            # Categorize
            dirs = []
            files_by_ext = {}
            others = []

            for item in items:
                full_item_path = os.path.join(path, item)
                if os.path.isdir(full_item_path):
                    dirs.append(item)
                else:
                    parts = item.rsplit('.', 1)
                    if len(parts) > 1 and parts[0]:
                        ext = parts[1].lower()
                        if ext not in files_by_ext:
                            files_by_ext[ext] = []
                        files_by_ext[ext].append(item)
                    else:
                        others.append(item)
            
            # 1. Directories
            if dirs:
                if len(dirs) < 5:
                    result_lines.append("--- Directories ---")
                    result_lines.extend(dirs)
                else:
                    result_lines.append(f"📂 Directories ({len(dirs)} items)")
            
            # 2. Files by extension
            for ext in sorted(files_by_ext.keys()):
                f_list = files_by_ext[ext]
                if len(f_list) < 5:
                    result_lines.append(f"--- Files (.{ext}) ---")
                    result_lines.extend(f_list)
                else:
                    result_lines.append(f"📦 *.{ext} ({len(f_list)} files)")
            
            # 3. Others
            if others:
                if len(others) < 5:
                    result_lines.append("--- Other Files ---")
                    result_lines.extend(others)
                else:
                    result_lines.append(f"📄 Other files ({len(others)} items)")

        else:
            result_lines.append("---")
            result_lines.extend(items)

        return "\n".join(result_lines)


class ReadFileTool(BaseTool):
    """
    安全版文件读取工具。

    自动截断大文件，对代码文件有更高的截断阈值。
    """
    name: str = "read_file"
    description: str = (
        "Read the content of a file. Large files will be truncated. "
        "Python files (.py) have a higher truncation limit. "
        "Input: the absolute path to a file."
    )
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, file_path: str) -> str:
        """读取文件内容。"""
        if not os.path.exists(file_path):
            return f"[ERROR] File does not exist: {file_path}"
        if not os.path.isfile(file_path):
            return f"[ERROR] Path is not a file: {file_path}"

        try:
            size = os.path.getsize(file_path)
        except Exception as e:
            return f"[ERROR] Cannot get file size: {e}"

        # 根据文件类型决定截断阈值
        is_code_file = file_path.endswith((".py", ".js", ".ts", ".java", ".c", ".cpp", ".h"))
        limit = MAX_READFILE_CHARS_CODE if is_code_file else MAX_READFILE_CHARS_DEFAULT

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(limit + 1)  # 读取多一个字符以判断是否截断
        except Exception as e:
            return f"[ERROR] Failed to read file: {e}"

        truncated = len(content) > limit
        if truncated:
            content = content[:limit]

        result_lines = [
            f"File: {file_path}",
            f"Size: {size} bytes",
        ]
        if truncated:
            result_lines.append(f"[TRUNCATED: showing first {limit} characters]")
        result_lines.append("--- Content Start ---")
        result_lines.append(content)
        result_lines.append("--- Content End ---")

        return "\n".join(result_lines)


class WriteFileTool(BaseTool):
    """
    写文件工具。

    完全覆盖目标文件。如果目录不存在会自动创建。
    对 solution.py 文件会自动创建快照。
    """
    name: str = "write_file"
    description: str = (
        "Write content to a file, overwriting if it exists. "
        "Parent directories will be created automatically. "
        "Input: path (absolute path) and content (the text to write)."
    )
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str) -> str:
        """写入文件。"""
        try:
            # 确保父目录存在
            parent_dir = os.path.dirname(path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # solution.py 快照逻辑
            if os.path.basename(path) == "solution.py":
                try:
                    snapshot_dir = os.path.join(parent_dir, ".snapshots")
                    os.makedirs(snapshot_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    snapshot_path = os.path.join(snapshot_dir, f"solution_{timestamp}.py")
                    with open(snapshot_path, "w", encoding="utf-8") as f:
                        f.write(content)
                except Exception:
                    pass  # 快照失败不影响主写入

            # 执行写入
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            bytes_written = len(content.encode("utf-8"))

            # 生成预览
            preview_limit = 500
            preview = content[:preview_limit]
            preview_truncated = len(content) > preview_limit

            result_lines = [
                f"[SUCCESS] File written: {path}",
                f"Bytes written: {bytes_written}",
            ]
            if preview_truncated:
                result_lines.append(f"[Preview truncated, showing first {preview_limit} chars]")
            result_lines.append("--- Preview ---")
            result_lines.append(preview)

            return "\n".join(result_lines)

        except Exception as e:
            return f"[ERROR] Failed to write file: {e}"


class TerminalTool(BaseTool):
    """
    Shell 命令执行工具。

    自动激活指定的 Conda 环境，并执行命令。
    成功时截断输出，失败时返回完整 stderr 以便调试。
    """
    name: str = "terminal"
    description: str = (
        "Execute a shell command in the specified Conda environment. "
        "Use this to run Python scripts, install packages, or perform system operations. "
        "The command will be executed in a bash shell with the Conda environment activated. "
        "Input: the command to execute."
    )
    args_schema: Type[BaseModel] = TerminalInput

    # 配置参数
    conda_env_name: str = ""
    default_timeout: int = 300
    long_running_timeout: int = 3600 * 2  # 2 小时

    def __init__(self, conda_env_name: str = "", **kwargs):
        """
        初始化 Terminal 工具。

        参数:
            conda_env_name: 要激活的 Conda 环境名称。
        """
        super().__init__(conda_env_name=conda_env_name, **kwargs)

    def _create_safe_bin(self) -> str:
        """
        创建包含非交互式 wrapper 脚本的安全 bin 目录。
        返回该目录的绝对路径。
        """
        safe_bin_dir = os.path.expanduser("~/.agent_safe_bin")
        os.makedirs(safe_bin_dir, exist_ok=True)

        # 定义 wrapper 脚本内容
        # 核心思想：强制添加 -o -f -y 等非交互参数
        wrappers = {
            "unzip": '#!/bin/bash\n/usr/bin/unzip -o -q "$@"',
            "cp": '#!/bin/bash\n/bin/cp -f "$@"',
            "mv": '#!/bin/bash\n/bin/mv -f "$@"',
            "rm": '#!/bin/bash\n/bin/rm -f "$@"',
        }

        for cmd, script_content in wrappers.items():
            wrapper_path = os.path.join(safe_bin_dir, cmd)
            # 仅在文件内容不同或不存在时写入，减少 IO
            write_needed = True
            if os.path.exists(wrapper_path):
                try:
                    with open(wrapper_path, "r") as f:
                        if f.read().strip() == script_content.strip():
                            write_needed = False
                except Exception:
                    pass
            
            if write_needed:
                try:
                    with open(wrapper_path, "w") as f:
                        f.write(script_content)
                    # chmod +x
                    st = os.stat(wrapper_path)
                    os.chmod(wrapper_path, st.st_mode | 0o111)
                except Exception:
                    pass  # 如果写入失败，尽力而为

        return safe_bin_dir

    def _run(self, command: str) -> str:
        """执行 Shell 命令。"""
        env_name = self.conda_env_name.strip()
        if not env_name:
            return "[ERROR] conda_env_name not configured. Cannot execute command."

        # 查找 conda 可执行文件
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        if shutil.which(conda_exe) is None:
            return f"[ERROR] Conda executable not found: {conda_exe}"

        # 准备 safe bin
        safe_bin_dir = self._create_safe_bin()

        # 决定超时时间
        # 仅对 solution.py 或 train.py 使用长超时
        is_long_running = "solution.py" in command or "train.py" in command
        timeout = self.long_running_timeout if is_long_running else self.default_timeout

        # 构建激活链
        # 注意：我们将 safe_bin_dir 添加到 PATH 的最前面，优先级最高
        activation_chain = (
            f'export PATH="{safe_bin_dir}:$PATH" '
            f'&& eval "$({conda_exe} shell.bash hook)" '
            f"&& conda activate {env_name} "
            f"&& {command}"
        )

        stdout_chunks = []
        stderr_chunks = []
        start_time = time.time()
        
        # 防御性初始化 PGID，防止 UnboundLocalError
        pgid = None
        
        try:
            # start_new_session=True 用于创建新的进程组
            # 这允许我们在超时时杀死整个进程组，防止僵尸管道
            proc = subprocess.Popen(
                ["/bin/bash", "-lc", activation_chain],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                start_new_session=True,
            )
            # 缓存 PGID，因为如果进程先退出被 poll() 回收，os.getpgid(proc.pid) 会失效
            pgid = proc.pid

            # 使用 select 进行非阻塞读取和活性检测
            # 定义宽限期：主进程退出后，最多再读 1 秒
            grace_period = 1.0
            exit_time = None
            
            while True:
                # 1. 检查超时
                if time.time() - start_time > timeout:
                    raise subprocess.TimeoutExpired(proc.args, timeout)
                
                # 2. 检查主进程状态
                return_code = proc.poll()
                
                # 3. 准备 select 列表
                reads = []
                if proc.stdout: reads.append(proc.stdout)
                if proc.stderr: reads.append(proc.stderr)
                
                # 如果没有可读的（例如管道已关），且进程已退出，结束
                if not reads and return_code is not None:
                    break

                # 4. 执行 select (设为较短超时以便频繁 check time/poll)
                try:
                    readable, _, _ = select.select(reads, [], [], 0.5)
                except ValueError:
                    # Select failed (possibly file descriptor closed), break loop if process done
                    if return_code is not None:
                         break
                    else:
                        continue # Retry

                # 5. 读取数据
                for f in readable:
                    try:
                        # 使用 os.read 读取底层 FD，确保非阻塞
                        # read() on TextIOWrapper 可能会即使 select 可读也阻塞（因为缓冲或解码）
                        fd = f.fileno()
                        b_chunk = os.read(fd, 4096)
                        
                        if not b_chunk:
                            # EOF received (empty bytes)
                            pass
                        else:
                            chunk = b_chunk.decode('utf-8', errors='replace')
                            if f is proc.stdout:
                                stdout_chunks.append(chunk)
                            else:
                                stderr_chunks.append(chunk)
                    except OSError:
                        pass
                    except Exception as e:
                       pass
                
                # 6. 退出条件判断
                if return_code is not None:
                    if exit_time is None:
                        exit_time = time.time()
                    
                    # 如果超过宽限期，强制退出循环
                    if time.time() - exit_time > grace_period:
                        break
                    
                    # 另外如果 readable 为空（管道已空/关闭），也退出
                    # 但为了保险（数据在内核缓冲），我们主要依赖宽限期或 EOF
                    if not readable:
                         pass
                    
            # 循环结束后，确保关闭管道防止 ResourceWarning
            if proc.stdout: proc.stdout.close()
            if proc.stderr: proc.stderr.close()
            
            # 手动清理：确保所有子进程（僵尸）都被杀掉
            # 无论成功失败，既然主任务结束了，就清理现场
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass

            exit_code = return_code if return_code is not None else -1
            stdout = "".join(stdout_chunks)
            stderr = "".join(stderr_chunks)

        except subprocess.TimeoutExpired:
            # 超时处理
            if pgid is not None:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass
            
            return (
                f"[ERROR] Command timed out after {timeout} seconds.\n"
                f"Command: {command}\n"
                f"Partial stdout:\n{self._truncate(''.join(stdout_chunks), MAX_STDOUT_CHARS)}\n"
                f"Partial stderr:\n{self._truncate(''.join(stderr_chunks), MAX_STDERR_CHARS)}"
            )
            
        except Exception as e:
            return f"[ERROR] Failed to execute command: {e}"

        # 处理结果
        if exit_code == 0:
            stdout_display = self._truncate(stdout, MAX_STDOUT_CHARS) if stdout else "(no output)"
            stderr_display = self._truncate(stderr, MAX_STDERR_CHARS) if stderr else ""

            result_lines = [
                f"[SUCCESS] Command completed (exit code: {exit_code})",
                f"Command: {command}",
            ]
            if len(stdout) > MAX_STDOUT_CHARS:
                result_lines.append(f"[stdout truncated to {MAX_STDOUT_CHARS} chars (head+tail)]")
            result_lines.append("--- stdout ---")
            result_lines.append(stdout_display)
            if stderr_display:
                result_lines.append("--- stderr ---")
                result_lines.append(stderr_display)

            return "\n".join(result_lines)
        else:
            result_lines = [
                f"[FAILED] Command failed (exit code: {exit_code})",
                f"Command: {command}",
                "--- stdout ---",
                self._truncate(stdout, MAX_STDOUT_CHARS) if stdout else "(no output)",
                "--- stderr (full/truncated) ---",
                self._truncate(stderr, MAX_STDERR_CHARS_FAILURE) if stderr else "(no error output)",
            ]
            # 交互式命令提示
            if "EOF" in (stderr or "") or exit_code != 0:
                result_lines.append(
                    "\nHint: If you see EOF errors, the command may require interactive input. "
                    "Use non-interactive flags like -y, -o, --yes, etc."
                )
            return "\n".join(result_lines)

    def _truncate(self, text: str, max_chars: int) -> str:
        """
        截断文本，保留头部和尾部。
        策略：保留 20% 头部，80% 尾部（尾部包含报错，权重更高）。
        """
        if not text or len(text) <= max_chars:
            return text
        
        head_len = int(max_chars * 0.2)
        tail_len = int(max_chars * 0.8)
        
        # 确保中间至少省略了一些内容，否则没必要截断
        if head_len + tail_len >= len(text):
           return text
           
        return (
            f"{text[:head_len]}\n"
            f"... [Output Truncated: omitted {len(text) - (head_len + tail_len)} chars] ...\n"
            f"{text[-tail_len:]}"
        )


# ============================================================================
# 工具加载函数
# ============================================================================
def get_tools(conda_env_name: str) -> list:
    """
    获取所有可用工具的列表。

    参数:
        conda_env_name: 要激活的 Conda 环境名称。

    返回:
        工具实例列表。
    """
    return [
        ListDirectoryTool(),
        ReadFileTool(),
        WriteFileTool(),
        TerminalTool(conda_env_name=conda_env_name),
    ]
