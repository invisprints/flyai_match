from flyai.train_helper import submit, upload_data, download, sava_train_model

"""""""""""""""""""""""""""
"         提交训练         "
"""""""""""""""""""""""""""
# train_name: 提交训练的名字,推荐使用英文，不要带特殊字符
# code_path: 提交训练的代码位置，不写就是当前代码目录，也可以上传zip文件
# cmd: 在服务器上要执行的命令，多个命令可以用 && 拼接
# 如：pip install -i https://pypi.flyai.com/simple keras && python train.py -e=10 -b=30 -lr=0.0003
# 会把当前submit所在的代码目录提交，cmd可以自己编写，GPU上使用python开头即可
submit("train_mnist", cmd="python train.py")

# 另一种提交方式，提交代码压缩包,目前支持zip格式的压缩包,代码会自动解压到运行目录下
submit("train_mnist", "D:/xxxxx.zip", cmd="python train.py")

"""""""""""""""""""""""""""
"        上传数据集        "
"""""""""""""""""""""""""""
# data_file:数据集的路径
# overwrite:模型名称相同的时候再上传是否覆盖，True会覆盖，False系统会重新命名
# 下载直接使用文件名称下载即可
# dir_name:文件夹名称，可以创建目录，用做斜线划分，目录不要有中文和特殊字符
# 例如:"/dataset" "/mydata/mnist"

upload_data("D:/data/MNIST.zip", overwrite=True)
# 上传之后在服务器上使用文件名下载数据集
# 服务器上数据下载地址为 ./MNIST.zip  decompression为True会自动解压
download("MNIST.zip", decompression=True)

# 或者设置路径上传数据，会自动在您的数据盘中创建路径
upload_data("D:/data/MNIST.zip", overwrite=True, dir_name="/data")
# 服务器上数据下载地址为 ./dataset/MNIST.zip  decompression为True会自动解压
download("/data/MNIST.zip", decompression=True)

"""""""""""""""""""""""""""
"     保存GPU上的模型      "
"""""""""""""""""""""""""""
# 上传自己的数据集
# model_file 模型在服务器上的路径加名字
# overwrite 是否覆盖上传
# dir_name 模型保存在数据盘中的目录
sava_train_model(model_file="./data/output/你的服务器上模型的名字", dir_name="/model", overwrite=False)

# 遇到问题不要着急，添加小姐姐微信
"""""""""""""""""""""""""""
"     小姐姐微信flyaixzs   "
"""""""""""""""""""""""""""

# 项目中遇到的依赖请写到requirements.txt中
