# 任务介绍
近几年视频媒体和直播行业的迅猛发展，使弹幕成为一种新潮的评论形式。用户观看视频的同时发表自己的想法和意见，并实时叠加在对应的视频页面上。弹幕技术冲击着传统主流媒体，让视频更加地趋于社交化，利用自然语言处理的相关技术对弹幕进行情感分析，在推荐系统、影视评分、热点管控以及节目优化等方面有极大的利用价值。而弹幕文本本身存在话题发散性、碎片化、对象多元化和语法的不完整性，要对这一特殊数据的情感进行研究，有一定的挑战空间。
BERT是一种预训练语言表示的方法，它优于传统的模型，是第一个用于预训练NLP的无监督，深度双向系统，此模型在NLP的多个上游下游问题上都取得很好的成绩。因此，本文将BERT预训练模型应用在弹幕数据的研究任务中，尝试提升文本情感分类的精准率。主要研究内容如下：
（1）从哔哩哔哩平台热门综艺视频爬取弹幕数据，提取用户、时间和文本等信息，进行去重、降噪、人工情感标注等数据预处理，处理成标准公开数据集。
（2）划分数据集，对比使用多个机器学习模型和BERT模型进行一个情感二分类任务的训练，使用测试集进行测试，通过对比F1值和召回率、精准率等指标衡量最优的分类模型。

# 1. BERT配置
要求配置如下, tensorflow版本不能太低。
```bush
tensorflow >= 1.11.0   # CPU Version of TensorFlow.
tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
```
我自己的版本是
```
Python 3.6.8 (default, May  7 2019, 14:58:50) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.__version__
'1.11.0'

```

## 1 数据处理
我们一共需要两种数据
- 数据是BERT开放的预训练模型
- 我们的数据集, 通过在bilibili平台综艺视频的爬虫获取，经过分词、去重、去停用词等预处理得到。
### 1.2 预训练模型
这是google花费大量**资源训练出来的预训练模型**， 我的数据集是英文句子， 所以我们使用 **BERT-base，uncased** 
也就是基础版本+有大小写版本， 当然如果你用中文，google也提供了相应的数据集

下载地址如下： [BERT-Base, Uncased下载地址](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
其他的可以在仓库的readme里找到相应的下载地址
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190517011805279.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxODc0NDU1OTUz,size_16,color_FFFFFF,t_70) 
然后随便放到一个地方，**这里放到了uncased文件夹里**， 路径不一样，模型运行参数会有一点点变化
### 1.2.2数据集
数据集介绍如下：

这是一个面向句子的情感分类问题。训练集和测试集已给出，使用训练集进行模型训练并对测试集中各句子进行情感预测。训练集包含10026行数据，测试集包含4850行数据。

#### 训练集
训练集中，每一行代表一条训练语句，包括四部分内容，使用'\t'分隔符：

#### 测试集
测试集中，每一行代表一条待预测语句，包括四部分内容，使用'\t'分隔符：

**需要从训练集里面挑10%作为开发集**

#### 开发集
开发集是由训练集得到的，我们使用pandas得到开发集， 代码如下, 开发集和训练集比例为9:1


这样我们就得到我们的数据集合。



# 2. 修改代码

因为这次是分类问题， 所以我们需要修改run_classify.py


## 2.1  加入新的处理类

因为我们是做一个分类的任务， 里面自带4个任务的处理类， 其中ColaProcessor是单句分类，和我们的任务较为相近， **所以我们模仿这个类，写一个自己的处理类。**

```py
class EmloProcessor(DataProcessor):
    """Processor for the Emotion data set ."""

    def get_train_examples(self, data_dir):
        """定义开发集的数据是什么，data_dir会作为参数传进去， 这里就是加上你的文件名即可 """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv"), ), "train")

    def get_dev_examples(self, data_dir):
        """定义开发集的数据是什么，data_dir会作为参数传进去，模型训练的时候会用到，这里就是加上你的文件名即可 """
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """定义测试集的数据是什么， 用于预测数据 ，在训练时没有用到这个函数， 这里写预测的数据集"""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """ 这里是显示你一共有几个分类标签， 在此任务中我有3个标签，如实写上  标签值和 csv里面存的值相同 """
        return ["neutral", "positive", "negative"]

    def _create_examples(self, lines, set_type):
        """这个函数是用来把数据处理， 把每一个例子分成3个部分，填入到InputExample的3个参数
        text_a 是 第一个句子的文本
        text_b 是 第二个句子的文本 但是由于此任务是单句分类， 所以 这里传入为None
        guid 是一个二元组  第一个表示此数据是什么数据集类型（train dev test） 第二个表示数据标号
        label 表示句子类别
        """
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)
            #print(line, i)
            # 获取text  第三列和第四列分别是 类别  文本 所以分情况添加
            text_a = tokenization.convert_to_unicode(line[3])
            if set_type == "test":
                #测试集的label 是要预测的 所以我们暂时先随便填一个类别即可 这里我选择都是neutral类
                label = "neutral"
            else:
                label = tokenization.convert_to_unicode(line[2])

            # 加入样本
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

```

有兴趣的盆友可以看看其他的3个类。


## 2.2 处理类注册

同样我们需要在主函数里把我们的类**当做参数选项**，给他加个选项， 也就是**当参数填emlo时，使用的数据处理类是我们自己写的处理类**
```py
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "emlo": EmloProcessor
  }

```



# 3. 运行代码

运行代码需要提供参数， 这里我建议**直接在pycharm编译器里加参数**，或者直接命令行运行参数， 而不用按照官方教材  run  xxx.sh

这里我给出我的编译参数， 如果你运行不了， 建议 改小max_seq_length， train_batch_size,
```shell
python
run_classifier.py
--task_name=emlo
--do_train=true
--do_eval=true
--data_dir=./glue
--vocab_file=./uncased/uncased_L-12_H-768_A-12/vocab.txt
--bert_config_file=./uncased/uncased_L-12_H-768_A-12/bert_config.json
--init_checkpoint=./uncased/uncased_L-12_H-768_A-12/bert_model.ckpt
--max_seq_length=128
--train_batch_size=32
--learning_rate=2e-5
--num_train_epochs=3.0
--output_dir=./tmp/emotion/
```

- task_name 表示我调用的是什么处理类 ，这里我们是用我们新的类所以选 emlo
- 文件dir 可以自己定义， 如果无定义到会出错， 我这里是有3个文件夹 uncased里面放预训练模型， glue放数据，tmp/emotion里面放结果

**训练结果如下：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190519002038639.png)


# 4. 分类预测
## 4.1 修改参数， 进行预测

预测的话， 将运行参数改为以下即可
```
python run_classifier.py 
  --task_name=emlo 
  --do_predict=true 
  --data_dir=./glue 
  --vocab_file=./uncased/uncased_L-12_H-768_A-12/vocab.txt 
  --bert_config_file=./uncased/uncased_L-12_H-768_A-12/bert_config.json 
  --init_checkpoint=./tmp/emotion/
  --max_seq_length=128 
  --output_dir=./tmp/emotion_out/
```
将会返回一个tsv， 每一列表示这一行的样本是这一类的概率
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190519002215733.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxODc0NDU1OTUz,size_16,color_FFFFFF,t_70)
每一类代表的类别和 在run_classify.py里面定义的lab顺序相同
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190519002304515.png)
## 4.2 得到类别
结果不能是概率，而是类别， 所以我们写一个脚本进行转化
**``get_results.py``**
```py
import os
import pandas as pd


if __name__ == '__main__':
    path = "tmp/emotion_out/"
    pd_all = pd.read_csv(os.path.join(path, "test_results.tsv") ,sep='\t',header=None)

    data = pd.DataFrame(columns=['polarity'])
    print(pd_all.shape)

    for index in pd_all.index:
        neutral_score = pd_all.loc[index].values[0]
        positive_score = pd_all.loc[index].values[1]
        negative_score = pd_all.loc[index].values[2]

        if max(neutral_score, positive_score, negative_score) == neutral_score:
            # data.append(pd.DataFrame([index, "neutral"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = ["neutral"]
        elif max(neutral_score, positive_score, negative_score) == positive_score:
            #data.append(pd.DataFrame([index, "positive"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = [ "positive"]
        else:
            #data.append(pd.DataFrame([index, "negative"],columns=['id','polarity']),ignore_index=True)
            data.loc[index+1] = [ "negative"]
        #print(negative_score, positive_score, negative_score)

    data.to_csv(os.path.join(path, "pre_sample.tsv"),sep = '\t')
    #print(data)
```
## 4.3 pf1.py评价
根据传入的文件true_label和predict_label来求模型预测的精度、召回率和F1值，另外给出微观和宏观取值。得到最终预测结果
