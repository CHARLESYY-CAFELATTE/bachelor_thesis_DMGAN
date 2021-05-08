# bachelor_thesis_DMGAN
一个简单的毕设小程序，用来做马赛克处理，不过效果并没有特别好，当然也有可能是我的数据集找的太烂了

训练输入于dataset/real，目标在dataset/construction

测试输入在input_image，目标在ideal_output

model分为有bias和没bias的，没bias的效果居然更好，针对我的训练集

image_SR用于图片还原，image_M能重新打码

train中不少参数都可以接受调整，继续训练也注意改改代码因为没记epoch用的还是变学习率的

感觉效果挺烂的，不过本科毕设要求没那么高

希望得到大佬指点，这只是新手的模型，应该很多地方都能调。

或者说这个问题好像本来就不可解......

用到的包：torch，pillow，opencv，numpy

这个代码好像没bug，python train.py就可以玩了，enjoy！

接口命名极其随意，开心就好哈哈哈

第一个自己试着做的算法，在github上发表以纪念一下

比论文里调薄了网络层，不过也能用

checkpoints权重没法上传，请移步链接: https://pan.baidu.com/s/14hxTervo6z6LkMsaZtbWxQ  密码: bpk2

放在根目录下就好，或者可以自己训练玩玩哈哈

以上
