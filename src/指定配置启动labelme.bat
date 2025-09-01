@echo off
setlocal

:: 替换 [..] 为你的 Conda 环境名称或完整路径
set "ENV_NAME=myenv"

:: 检查 Conda 是否可用
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到 Conda 环境。请确保 Anaconda/Miniconda 已安装并已添加至系统路径。
    pause
    exit /b 1
)

:: 激活 Conda 环境
call conda activate "%ENV_NAME%"
if %errorlevel% neq 0 (
    echo 错误: 无法激活环境 [%ENV_NAME%]。请检查环境名称是否正确。
    pause
    exit /b 1
)

:: 运行 labelme 命令
labelme --config .labelmerc --labels labels.txt --nodata

:: 如果 labelme 命令失败
if %errorlevel% neq 0 (
    echo 错误: labelme 执行失败。请检查配置文件和标签路径。
    pause
    exit /b 1
)

endlocal