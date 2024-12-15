REM 先安装 Node.js & go-task ！
REM Node.js: https://nodejs.cn/en/download/prebuilt-installer
REM npm config set registry https://registry.npmmirror.com
REM go-task: https://taskfile.dev/installation/
REM npm install -g @go-task/cli

REM ref: https://stackoverflow.com/questions/72142036/best-way-to-activate-my-conda-environment-for-a-python-script-on-a-windows-pc
REM 记得修改所使用的 conda 虚拟环境名称！
conda run -n ModelDemoWithStreamlitApp3 task dev
