# FROM registry.corp.kuaishou.com/kml-ad/env_of_maskrcnn_with_kess:10.1-cudnn7-devel-ubuntu16.04-hdfs_yunjiezhang_snapshot-9127-20221010212024
FROM registry.corp.kuaishou.com/kml-ad/env_of_maskrcnn_with_kess:10.1-cudnn7-devel-ubuntu16.04_202010282034-snapshot
# 拉取代码, 拉取环境，拉取文件
COPY . /home/web_server/live-cut-tool
RUN ls /root/datas/ && ls /home/web_server/live-cut-tool/
RUN cp /root/datas/pytorch_model.bin /home/web_server/live-cut-tool/final_code/bert/ \
&& cp /root/datas/sgns.sogou.char /home/web_server/live-cut-tool/final_code/word/

# 指定工作目录
WORKDIR /home/web_server/live-cut-tool/final_code
#RUN cd final_code \
#&& yum install -y epel-release \
#&& yum install -y make gcc gcc-c++ libcurl-devel openssl* unzip nginx crontabs mysql-devel \
#&& export CFLAGS="-std=c99" \
#&& pip3 install Django python-memcached mysqlclient requests uwsgi xlrd python-cas freetype-py fontTools brotli protobuf grpcio grpcio-tools \
#&& echo Asia/Shanghai > /etc/timezone

# && pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \

# && python3 -m grpc_tools.protoc -I "./proto" --python_out="." --grpc_python_out="." ./proto/*.proto

#ENV PYTHONPATH=.

#ENV LD_LIBRARY_PATH=/home/web_server/cuda-10.1-cudnn-7.6.0/lib64

# 安装依赖, 编译
# RUN chmod 777 final_code/build.sh && sh final_code/build.sh
RUN echo Asia/Shanghai > /etc/timezone
# 启动命令
CMD sh final_code/start.sh
