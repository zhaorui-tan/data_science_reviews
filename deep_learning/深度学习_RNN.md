[TOC]

RNN，又名循环神经网络，是现在主流的神经网络单元之一，十分适合序列模型如文本等，其衍生的诸多变体在自然语言处理领域应用广泛，其中LSTM更是在NLP领域鼎鼎大名。

# 基本的RNN


```
graph LR
x--U-->id2((s))
id2((s))--W-->id2((s))
id2((s))--V-->id3((o))
```

- x ： 表示输入层的值。
- U：表示输入层到隐层的权重矩阵
- s ：表示隐层的输出值
- V：表示隐层到输出层的权重矩阵
- o ： 表示输出层的值

展开如下
```
graph BT
subgraph 
s_t+1--V-->o_t+1
x_t+1--U-->s_t+1
end
subgraph 
x_t--U-->s_t
s_t--V-->o_t
s_t--W-->s_t+1
end
subgraph 
x_t-1--U-->s_t-1
s_t-1--V-->o_t-1
s_t-1--W-->s_t
end

```

- x_t ： 表示 t 时刻的输入， 通常为向量， 在NLP中， 常常将词向量作为RNN的输入。
- s_t ： 表示 t 时刻的隐层输出。
- o_t ： 表示 t 时刻的输出层输出。
- 
## W 是什么？
在基本的RNN模型中，某一时刻隐层的值 s_t 不仅仅取决于当前的输入 x_t ，还取决于上一次的隐层的值 s_t-1。而 W 就是隐层上一时刻的值对当前时刻隐层输出的影响权重。

## 公式

```math
输出层计算公式： o_t = g(V_{s_t}) 

隐层计算公式： s_t = f(U_{x_t}+W_{s_{t-1}}) 

其中， g() f() 表示激活函数。
```
我们将循环神经网络与前馈神经网络比较起来会发现，二者之间只是多了一个反馈的过程。结合上述两式可以得出输出层的输出为：

```math
o_t =  g(V_{s_t}) 

= Vf(U_{x_t}+W_{s_{t-1}})

= Vf(U_{x_t}+Wf(U_{x_{t-1}}+W_{s_{t-2}}))


= Vf(U_{x_t}+Wf(U_{x_{t-1}}+Wf(U_{x_{t-2}}+W_{s_{t-3}})))
```


RNN中的输出值o_t 是受到前面历次输入值影响的，这也是为什么RNN在NLP领域中广泛应用的原因。


## 几种不同类型的RNN
根据输入与输出的序列对比可以将RNN简单分为四种类型： N vs N， N vs 1， 1 vs N， N vs M。

###  N vs N
这是一个很经典的结构，该结构表示： 输入和输出序列必须要是等长的。如下图所示：

展开如下
```
graph BT
subgraph 
x3-->h3
h3-->y3
end
subgraph 
x2-->h2
h2-->y2
h2-->h3
end
subgraph 
x1-->h1
h1-->y1
h1-->h2
end
subgraph 
h0-->h1
end
```

通俗来讲，如果我有n个输入，那么我就会有n个输出。这个结构跟上面基本RNN网络是一样的。

这个结构要求输入序列与输出序列必须等长，这就产生了一些限制，比如机器翻译中两种语言翻译过后其实并不等长等。由于这个由于这个限制的存在，**经典RNN的适用范围比较小**，但也有一些问题适合用经典的RNN结构建模，如：

- 计算视频中每一帧的分类标签。因为要对每一帧进行计算，因此输入和输出序列等长。


### N vs 1
有时候，我们输入的是一个序列，但是我们的输出是一个单独的值。比如文本情感分类任务，举个例子如下。

例子：

我觉得这部电影很好看

分析：

我们要对这句话的情感进行分析，它是正向的还是消极的？那么我吗最终输出的就是一个标签（label）。
如上面举例而言，该结构很适合序列分类问题。

 Y = Softmax(Vh_t + c)
```
graph BT 
subgraph 
x3-->h3
h3-->Y
end
subgraph 
x2-->h2
h2-->h3
end
subgraph 
x1-->h1
h1-->h2
end
subgraph 
h0-->h1
end

```

### 1 vs N
该结构输入的是一个值，而输出的是一个序列。最常见的例子是通过图像生成文字：X表示图像的特征，输出的y序列表示对这段图像的一段描述。

#### 常见的有以下两种结构：

只在序列开始处进行输入计算

```
graph BT 
subgraph 
h3-->y3
end
subgraph 
h2-->h3
h2-->y2
end
subgraph 
h1-->y1
X-->h1
h1-->h2
end
subgraph 
h0-->h1
end

```

将输入信息X作为每个阶段的输入

```
graph BT 
subgraph 
h3-->y3
end
subgraph 
h2-->h3
h2-->y2
end
subgraph 
h1-->y1
h1-->h2
end
subgraph 
X-->h1
X-->h2
X-->h3
h0-->h1
end

```

###  N vs M
这应该是这四种变体中最常见也最重要的一个变体 -- 输入一组长度为N的序列，产生长度为M的序列。最常见的是在机器翻译中，因为两种语言对于相同意思的表达，其长度可能是不同的。

**==N vs M结构又叫Encoder-Decoder模型，也可以称之为Seq2Seq模型==。**

Encoder-Decoder结构先将输入数据编码成一个上下文向量c，这个过程就叫做Encoding，得到c的方法有多种，这里只讲结构，不涉及具体模型。拿到c之后，就用另一个RNN网络对其进行解码，这部分RNN网络被称为Decoder。

在Decoder部分，常见有两种做法：

将上下文向量c当做之前的初始状态输入到Decoder中，如下图所示：

```
graph BT 
subgraph 
h3'-->y3
end
subgraph 
h2'-->y2
h2'-->h3'
end
subgraph 
h1'-->y1
h1'-->h2'
end
subgraph 
C-->h1'
end
subgraph 
x3-->h3
h3-->C
end
subgraph 
x2-->h2
h2-->h3
end
subgraph 
x1-->h1
h1-->h2
end
subgraph 
h0-->h1
end
```

另一种做法是将c作为每一步的输入：

```
graph BT 
subgraph 
h3'-->y3
end
subgraph 
h2'-->y2
end
subgraph 
h1'-->y1
end
subgraph 
h0'-->h1'
end
subgraph 
C-->h1'
C-->h2'
C-->h3'
end
subgraph 
x3-->h3
h3-->C
end
subgraph 
x2-->h2
h2-->h3
end
subgraph 
x1-->h1
h1-->h2
end
subgraph 
h0-->h1
end
```
### 双向RNN
仔细分析普通的RNN模型，我们发现它有一个缺陷：它只能记住前面的输入对它的影响，而不能将后面的输入对它的影响记忆下来。

针对上面的这个问题，产生了双向RNN，其模型如下：

```
graph BT 
subgraph 
S0-->A0
A0-->A1
A1-->A2
A2-->Ai
Ai-->S1
end

subgraph 
S0'-->Ai'
Ai'-->A2'
A2'-->A1'
A1'-->A0'
A0'-->S1'
end

subgraph 
x0-->A0
x1-->A1
x2-->A2
xi-->Ai
end

subgraph 
xi-->Ai'
x2-->A2'
x1-->A1'
x0-->A0'
end

subgraph 
A0-->y0
A1--> y1
A2--> y2
Ai--> yi
end

subgraph 
Ai'-->yi
A2'-->y2
A1'-->y1
A0'-->y0
end
```
将一段序列反向再来一遍基本的RNN模型，也就是说，双向RNN其实是正向RNN + 反向RNN。

我们来通过计算一下 y2 的值理解该模型。双向RNN的隐藏层中要保存两个值：

- A： 参与正向运算
- A'： 参与反向运算


```math
y_2 =  g(VA_2 + V'A_2')
```
g() ： 表示激活函数
V ： 表示 A_2到输出值 y_2 的权重矩阵
V'： 表示A_2' 到输出值 y2' 的权重矩阵

### 注意：
- 在双向RNN中，正向计算与反向计算不共享权重。
- 双向RNN中的每个单元可以是普通RNN单元，也可以是LSTM单元，GRU单元等。

# 深度RNN
在深度循环神经网络中，每个单元都可以是一个普通的RNN单元，LSTM单元或者GRU单元。


# 最后
RNN的突破在于思想， 而在实际的NLP任务中， 通常由于文本的长度较长使得RNN的深度较深， 很容易产生梯度消失，梯度爆炸问题， 于是基于门机制的 LSTM，GRU诞生了。 在现有的paper中，几乎看不到使用原始的RNN来作为基本单元，所以， LSTM才是我们要真正要熟练掌握的。

