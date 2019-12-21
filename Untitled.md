##### step 0: preprocessing


```python
import fastNLP.io

#获取数据集地址
Loader = fastNLP.io.loader.IMDBLoader()
str = Loader.download(dev_ratio=0.1, re_download=False)

```


```python
#获取databundle
data_bundle = Loader.load(str)
Pipe = fastNLP.io.pipe.IMDBPipe()
data_bundle = Pipe.process(data_bundle)
print(data_bundle)

```

    In total 3 datasets:
    	dev has 633 instances.
    	train has 5746 instances.
    	test has 25000 instances.
    In total 2 vocabs:
    	words has 117134 entries.
    	target has 2 entries.
    



```python
#获取dataset
tr_data = data_bundle.get_dataset('train')
te_data = data_bundle.get_dataset('test')
print(tr_data)
print(te_data)
```

    +------------------------+--------+------------------------+---------+
    | raw_words              | target | words                  | seq_len |
    +------------------------+--------+------------------------+---------+
    | Yes, the cameras we... | 0      | [778, 4, 2, 4584, 7... | 184     |
    | This movie is total... | 0      | [59, 21, 9, 479, 43... | 43      |
    | You can never have ... | 1      | [214, 69, 136, 34, ... | 230     |
    | The Hazing is confu... | 1      | [19, 31192, 9, 1511... | 239     |
    | As a Dane I'm proud... | 1      | [221, 5, 16101, 11,... | 581     |
    | But it does have so... | 1      | [118, 12, 87, 34, 6... | 237     |
    | All I can say after... | 1      | [337, 11, 69, 152, ... | 160     |
    | Your mind will not ... | 1      | [2624, 354, 99, 33,... | 166     |
    | This is the finest ... | 0      | [59, 9, 2, 2087, 25... | 75      |
    | To my surprise, I r... | 0      | [431, 83, 877, 4, 1... | 178     |
    | Note: I will reveal... | 1      | [4798, 90, 11, 99, ... | 226     |
    | First off, I would ... | 1      | [684, 138, 4, 11, 6... | 211     |
    | ...                    | ...    | ...                    | ...     |
    +------------------------+--------+------------------------+---------+
    +------------------------+--------+------------------------+---------+
    | raw_words              | target | words                  | seq_len |
    +------------------------+--------+------------------------+---------+
    | Alan Rickman & Emma... | 1      | [1915, 6808, 196, 4... | 121     |
    | I have seen this mo... | 1      | [11, 34, 121, 15, 2... | 133     |
    | In Los Angeles, the... | 1      | [140, 2929, 3378, 4... | 203     |
    | This film is bundle... | 1      | [59, 25, 9, 28361, ... | 543     |
    | I only comment on r... | 1      | [11, 77, 923, 28, 7... | 241     |
    | When you look at th... | 1      | [288, 31, 186, 40, ... | 235     |
    | Rollerskating vampi... | 1      | [76711, 2740, 57, 3... | 167     |
    | Technically abomina... | 1      | [6102, 12561, 30, 2... | 95      |
    | When Hollywood is t... | 1      | [288, 365, 9, 281, ... | 339     |
    | Respected western a... | 1      | [76716, 1331, 7669,... | 280     |
    | Worst movie ever se... | 1      | [4179, 21, 144, 121... | 143     |
    | I was forced to wat... | 1      | [11, 18, 899, 8, 12... | 162     |
    | ...                    | ...    | ...                    | ...     |
    +------------------------+--------+------------------------+---------+



```python
#获取vocabulary
words_vocab = data_bundle.get_vocab('words')
print(words_vocab)
```

    Vocabulary(['Yes', ',', 'the', 'cameras', 'were']...)



```python
#获取特定word的index
index = words_vocab.to_index('why')
print(index)
```

    189


##### step 1: embedding


```python
from fastNLP.embeddings import StaticEmbedding

#glove
glove_embed = StaticEmbedding(words_vocab, model_dir_or_name='en-glove-6b-50d')
```

    Found 46289 out of 117134 words in the pre-training embedding.


#### step 2: build a model


```python
from fastNLP.models import CNNText
model = CNNText(glove_embed,2, kernel_nums=(30, 40, 50), kernel_sizes=(1, 3, 5), dropout=0.5)
```

#### step 3: train a model


```python
from fastNLP import Trainer
from fastNLP import CrossEntropyLoss
from torch.optim import Adam
from fastNLP import AccuracyMetric

loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
metric = AccuracyMetric()

trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, loss=loss,
                  optimizer=optimizer, batch_size=32, dev_data=data_bundle.get_dataset('dev'),
                  metrics=metric)
trainer.train()  # 开始训练，训练完成之后默认会加载在dev上表现最好的模型

# 在测试集上测试一下模型的性能
from fastNLP import Tester
print("Performance on test is:")
tester = Tester(data=data_bundle.get_dataset('test'), model=model, metrics=metric, batch_size=64)
tester.test()
```

    input fields after batch(if batch size is 2):
    	words: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2, 184]) 
    	seq_len: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
    target fields after batch(if batch size is 2):
    	target: (1)type:torch.Tensor (2)dtype:torch.int64, (3)shape:torch.Size([2]) 
    
    training epochs started 2019-12-21-15-05-57



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=1800.0), HTML(value='')), layout=Layout(d…



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.51 seconds!
    Evaluation on dev at Epoch 1/10. Step:180/1800: 
    AccuracyMetric: acc=0.821485
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.39 seconds!
    Evaluation on dev at Epoch 2/10. Step:360/1800: 
    AccuracyMetric: acc=0.815166
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.41 seconds!
    Evaluation on dev at Epoch 3/10. Step:540/1800: 
    AccuracyMetric: acc=0.827804
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.4 seconds!
    Evaluation on dev at Epoch 4/10. Step:720/1800: 
    AccuracyMetric: acc=0.827804
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.32 seconds!
    Evaluation on dev at Epoch 5/10. Step:900/1800: 
    AccuracyMetric: acc=0.834123
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.4 seconds!
    Evaluation on dev at Epoch 6/10. Step:1080/1800: 
    AccuracyMetric: acc=0.821485
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.61 seconds!
    Evaluation on dev at Epoch 7/10. Step:1260/1800: 
    AccuracyMetric: acc=0.827804
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.4 seconds!
    Evaluation on dev at Epoch 8/10. Step:1440/1800: 
    AccuracyMetric: acc=0.824645
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.62 seconds!
    Evaluation on dev at Epoch 9/10. Step:1620/1800: 
    AccuracyMetric: acc=0.826224
    



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=20.0), HTML(value='')), layout=Layout(dis…


    Evaluate data in 1.35 seconds!
    Evaluation on dev at Epoch 10/10. Step:1800/1800: 
    AccuracyMetric: acc=0.835703
    
    
    In Epoch:10/Step:1800, got best dev performance:
    AccuracyMetric: acc=0.835703
    Reloaded the best model.
    Performance on test is:



    HBox(children=(FloatProgress(value=0.0, layout=Layout(flex='2'), max=391.0), HTML(value='')), layout=Layout(di…


    Evaluate data in 71.99 seconds!
    [tester] 
    AccuracyMetric: acc=0.8374





    {'AccuracyMetric': {'acc': 0.8374}}




```python

```
