from flyai.train_helper import submit, upload_data

# 遇到问题不要着急，添加小姐姐微信
"""""""""""""""""""""""""""
"     小姐姐微信flyaixzs   "
"""""""""""""""""""""""""""
# train_name: 提交训练的名字,推荐使用英文，不要带特殊字符
# code_path: 提交训练的代码位置，不写就是当前代码目录，也可以上传zip文件
# cmd: 在服务器上要执行的命令，多个命令可以用 && 拼接
# 如：pip install -i https://pypi.flyai.com/simple keras && python train.py -e=10 -b=30 -lr=0.0003

# 把本地数据上传到自己的数据盘
upload_data("./data/MNIST.zip", dir_name="/data", overwrite=True)
# 会把当前submit所在的代码目录提交，cmd可以自己编写，GPU上使用python开头即可
# 系统设定不会提交data文件夹
submit("train_mnist", cmd="python train.py")
