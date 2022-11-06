# shellcheck disable=SC1113
#/bin/sh
#程序启动脚本
#注意一定要保持前台运行，不然docker会退出
export http_proxy="http://10.28.121.13:11080"
export https_proxy="http://10.28.121.13:11080"
export PYTHONPATH=/home/web_server/live-cut-tool/final_code

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

eval `ssh-agent -s`
ssh-add ~/.ssh/kwai-key

#export http_proxy=http://bjm7-squid4.jxq:11080 && export https_proxy=http://bjm7-squid4.jxq:11080


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/home/hadoop/software/hadoop/lib/native:/home/hadoop/software/hadoop/lib/native:/home/hadoop/software/java//jre/lib/amd64/server:/usr/TensorRT-5.1.5.0/lib

conda activate lzh
pip install datetime -i https://mirrors.aliyun.com/pypi/simple/
#pip install utils -i https://mirrors.aliyun.com/pypi/simple/
#pip install random -i https://mirrors.aliyun.com/pypi/simple/
python /home/web_server/live-cut-tool/final_code/main.py
#python /home/web_server/live-cut-tool/final_code/test_sent.py
#kcsize需要的环境变量
#wiki:https://wiki.corp.kuaishou.com/pages/viewpage.action?pageId=84107652
#export SUPERVISOR_PROGRAM0=grpc-demo-python
#conda activate lzh
#export SUPERVISOR_COMMAND0="python3 /home/web_server/live-cut-tool/live-cut-offline-tool/final_code/main.py"

#通过kcsize启动supervisor，supervisor托管服务进程
#kcsize supervisor
