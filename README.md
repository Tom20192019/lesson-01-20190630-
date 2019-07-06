# lesson-01 Assignment 如下： 

1 复现课堂代码 

noun_phrase => Article Adj* noun
Adj* => null | Adj Adj*
verb_phase => verb noun_phrase 
Article => 一个|这个
noun => 女人|篮球|桌子|小猫
verb => 看着|坐在|听着|看见
Adj => 蓝色的|好看的|小小的"""

simplest_grammar ="""
number = number number | single_number
single_number = 1|2|3|4|5|6|7|8|9|0
"""
import random
def adj(): return random.choice("蓝色的|好看的|小小的").split("|)).split()[0]
def adj_star(): return random.choice([None,adj() + adj()])
adj_star()

another_grammar ="""
Adj* = => null|Adj Adj*
Adj => 蓝色的|好看的|小小的"""      #turminal 
""" 
grammar={}
for line in adj_grammar.split("\n"):
  if not line.strip(): continue 
  exp,stmt = line.split("=>")
  grammar[exp.strip()] = [s.split() for s in stmt.split("|")]
 grammar ["Adj"]
 
 def generate(gram,target):
      if target not in gram:  return target  # means target is a terminal expression
      if target in gram:  # target could be expanded 
        new_expanded = random.choice(gram[target])
        reurn "".join(generate(gram,t) for t in new_expanded)
       else:
        retun target
        
 generate(gram=grammar,target="Adj*")
 
 def generate(gram,target):
     if target not in gram : return target 
     retunr "".join(generate(gram,t) for t in random.choice(gram[target]))
 generate(gram=grammar,target="Adj"
 
 
choice = random.choice
  def generate(gram,target):
    if target not in gram: return target 
    edxpand = [generate (gram,t) for t in choice (gram[gram[target])]
    return "".join ([e if e !="/n" else "\n" for e in expand if e ! "null"])
    
example_grammer = create_grammar(simple_grammar)
example_grammar
generate(gram=example_grammar,target="sentence" 

#在西部世界里，一个”人类“的语言可以定义为：

human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 找找 | 想找点 
活动 = 乐子 | 玩的
"""


#一个“接待员”的语言可以定义为

host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = null
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？
"""

for i in range(20):
print(generate(gram=create_grammar(host,split="="),target ="host"))


Data Driven

programming="""
stmt => if_exp|while_exp|assignment 
assignment => var=var 
if_exp=> if(var){/n.....stmt}
while_exp=> while(var) {/n ...stmt}
var => chars munber 
chars=> char\char char 
char => student|name|info|database|course
number =>1|2|3
"""
print(generate(gram=create_grammar(programming,split="=>"),target="stmt"))

Language Model

 
finename = ""
import pandas as pd
content = pd.read_csv(filename,encoding="gb18030)
content.head()
articles = content["content"].tolist
 
len(articles)
articles[110]

import re 
def token(string):
   return re.findall("\w+",string)
   "".join(token(articles[110] 
   
from collections import Counter
with_jieba_cut = Counter(jieba.cut(articles[110]))
wiht_jieba_cut.most_common[:10]
"".join(token(articles[110]))
articles_clean = ["".join(token(str(a))) for a in articles]
len(articles_clean)
with open("article_9k.txt","w") as f:
for a in articles_clean:
f.write(a+"\n")

def cut(string): return list (jieba.cut(string))
import jieba 
def cut(string): return list (jieba.cut(string))
TOKEN=[]
for i , line in enumerate((open("articles_9k.txt"）））：
  ifi% 100==0： print(i)
  if i > 10000: break
  TOKEN+= cut(line)
  
  from functools inport reduce 
  from operator import add, mul
  redunce(add,[1,2,3,4,5,8]
  [1,2,3,]+[3,43,5]
  
  from collections imprt Counter 
  words_count = Counter(TOKEN)
  words_count.most_commom(100)
  
  frenquiences = [ f doe w, f in words_count.most_common(100)]
  x = [i for i in range(100)]
  %matplotlib inline 
  inport matplotlib.pyplot as plt 
  plt.plt(x, frequiences)
  
  import numpy as np 
  plt.plot(x, np.log(frequiences))
  
  def prob_1(word):
    return words_count[word] / len(TOKEN) 
  prob_1("我们")
  TOKEN[:10]
  TOKEN = [str(t) for t in TOKEN 
  TOKEN_2_GRAM = ["".join)TOKEN[i:i+2] for i in range(len(TOKEN[:-2]))]
  TOKEN_2_GRAM[:10]
  words_count_2 = Counter(TOKEN_2_FRAM)
  def prob_1(word) : return wors_count[word] / len(TOKEN)
  
  def prob_2(word1,word2):
  if word1 + word2 in words_count_2: return words_count_2[word1+word2] / [len(TOKEN_2_GRAM)
  else:
  retun 1 / len(TOKEN_2_GRAM)
  
  prob_2("我们","在")
  prob_2("在","吃饭")
  prob_2("去","吃饭")
  
  def get_probablity(sentence):
  words =  cut(sentence)
  sentence_pro = 1 
  for in ,word in enumerate(words[:-1]):
  next_ = words[i+1] 
  probablity = prob_2(wors,next_)
  sentenxe_prob * = probablity 
  retunr sentence_pro 
  
  get_probablity("小明今天抽奖抽到一台苹果手机")
  get_probablity("小明今天抽奖抽到一架波音飞机")
  
  get_probablity("洋葱奶昔来一杯")
  get_probablity("养乐多绿来一杯")
  
  for sen in [generate(gram = example_grammar, target="sentence") for i in range(10):
  print("sentence:{} with Pro: {}.format(sen,get_probablity(sen))
  
  need_compared = [
      "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"]
    
    for s in need_compared:
    s1, s2 = s.split()
    p1,p2 = get_probablity(s1),get_probablity(s2)
    better = s1 if p1 > p2 else s2 
    
    print("{} is more possible".format(better))
    print("-"*4+" {} with probablity {}".format(s1,p1))
    print("-"*4+" {} wiht probablity {}".format(s1,p2))
    
    
    
    Language Model
      filename =("C:/Users/tyx51/file/train.txt")
import pandas as pd
content = pd.read_csv("train.txt",sep="\t")
content.head()
articles  = content["cotent"].tolist
len(articles)
import re
def token(string):
  return re.findall("\W+",string)
  from collection import Counter 
  with jieba_cut = Counter(jieba.cut9(articles[110]))
  with jiaba_cut.most_common()[:10]
  "".join(token(articels[110])
  artilcles_clean = ["".join(token(str(a) ))) for a in articles]
  len(articles)
  with open("articles_9k.txt","w") as f:
    for a in articles_clean:
    f.write(a+"\n")
def cut(string): return list(jieba.cut(string))
import jieab 
def cut(string):return list(jieba.cut(string)
TOKEN = []
for i ,line in enumerate ((open("articles_9k.txt")))
  if i % 100 == 0: print(i)
  if i > 10000: break 
  TOKEN+= cut(line)
  from functools import Counter 
  from operator import add,mul
  reduce(add,[1,2,3,4,5,8])
  from collections import Conter
  words_count.most_commom 
  frequiences = [f for w, f in words_count.most_common(100)]
  x= [i  for i in range(100)]
  %matplotlib inline 
  import matplotlib.pylot as plt 
  plt.plot(x,frequiences)
  import nunmoy as np 
  plt.plot(x,np.log(frequiences))
  def prob_1(word):
    return words_count[word]/ len(TOKEN)
    prob_1("我们”）
    TOKEN[:10]
    TOKEN = [str(t) for t in TOKEN]
    KOKEN_2_GRAM = ["".join(TOKEN[:i:i+2] for i in range(len(TOKEN[:-2]))]
    TOKEN_2_GRAM[:10]
    words_count_2 = Counter(TOKEN_2_GRAM)
    def prob_1(word): return words_count[word] / len(TOKEN)
    def prob_2(word1,word2):
    if word1 + word2 in word_count_2: reurn words_count_2[word1+word2] / len(TOKEN_2_GRAM)
    else:
    return 1 / len(TOKEN_2_GRAM)
    prob_2("我们","在")
    prob_2("在","吃饭")
    prob_2("去","吃饭")
    def get_probability(sentence):
    words = cut(sentence)
    sentence_pro = 1
    for i, word in enumarate(word,next_)
    sentence_pro * = probablity 
    return sentence_pro
    get_probablity("小明今天抽奖抽到一台苹果手机")
    get_probablity("小明今天抽奖抽到一架波音飞机")
    get_probablity("洋葱奶昔来一杯")
    get_probablity("养乐多绿来一杯")
    
    for sen in [generate(gram=example_grammar, target="sentence") for i in range(10)]:
      print("sentence: {} wiht Pro: {}".fomat(sen,get_probablity(sen)))
    need_compared = [
        "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"
      ] 
      for s in need_compared:
      s1,s2 = s.split()
      p1,p2 = get_probablity(s1),get_probablity(s2)
      better = s1 if p1 > p2 else s2 
      print("{} idmore possible".format (better))
      print("-"*4+' {} with probablity {}'.format(s1,p1))
      print("-"*4+' {} with probablity {}'.format(s1,p2)
  
  
  
  5. 完成以下问答和编程练习
  
  5.1 回答问题
  
  0. Can you come up out 3 sceneraies which use AI mehods?  
  medical, auot-drive , education, sale, home automation
  
  1. How do we use Github; Why do we use Jupyter and Pycharm;
  1.1 create a github repo
  A repo is storge space where you project lives. it can be local to a folder on your computer , or it can be a storage space on github or anther online host.
  1.2 create branches and perform operations 
  Branchings can help you to work on different version of a repository at one time. 
  how to use github :operations 
  commit command 
  now let’s make our first commit, follow the below steps:
  click on "read me -chages" file which  we have just created.
  click on the "edit" or a pencil icon in the rightmost cornoer of hte file.
  once you click on that , an editor will open where you can type in the changes ro anything.
  write a commit message whcih identifies your changes.
  click commit changes in the end .
  1.3 pull command 
  pull command is the most important command in guthub, it tell the chages done in the file and request other contributors to view it as well as merget it with the master branch. 
  1.4 merge command 
  here comes the last command which merge the chnages into the main master branch. 
  1.5 cloning and forking github repository 
  cloning: Suppose you want to use some code which is present in a public repository , you can directly copy the contets by clong or downloading. refer to use the below screccsht for a better understanding . 
  ForkingL Suppose,, you need some code which is present in a public repository , underyour repository and github account. 
  for this, we need to fork a repository. 
  
  why do we use jupyter and Pycharm?
  Jupyter is the three things: a collection of standars, a community, and a set of software tools. jupyter, is software that xreates a jupyter, is a document that supports mixing executable code, q=equations, visualization, and narrative text. Specially , jupyrt allow the user to bring together data, code and prose, to tell an interactive,computational story. 
  
  why do we use pycharm? 
  ok. here there are reasons to consider making Pycharm your primary editor to write Python code.
  1) code completion  pycharm has great code completion whether it;s for a built-in or an external package.
  2) prdfessional tools
  3) full-stack web develoment
  4) scientific tools 
  5) customizable & cross-platform 
  
  
  2. What's the Probablity Model?
  
  3. Can you came up wiht some sceneraies at which we use Probablity Model？
  
  4. Why do we use probablity and what's the difficult points for programming based on parsing and pattern match?
  
  5. What's the Language Model?
  
  6. Can you came up with some sceneraies at which we could use Language Model?
  7. What's the 1-gram language model ?
  
  8. what's the disadvanges and advantages of 1-gram language model？
  
  9. what's the 2-gram model?
  
  
  
  
  
  
  5.2 编程实践部分

  1. 设计自己的句子生成器
  #一个客户在4S店售车现场，咨询汽车信息的场景可以定义为：
human = """
human = 客户  咨询  车辆信息
客户 = 我 | 俺 | 我们 
咨询 = 想问问 | 想请问 | 请教您 | 方便问一下| 问下工程师| 打听下|能问下
车辆信息 = 关于D60车子的上市时间 | G90的颜色有哪几种 |这个车子百公里的油耗|大通房车的价格多少啊|
"""
#一个“智能机器人”的语言可以定义为

bot = """
bot = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号销售经理 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 了解关于车子的哪些信息 | 想咨询车子哪方面的内容呢| 询问哪一款车子
车辆信息 = 价格 | 颜色 | 油耗 | 上市时间
结尾 =？
"""
 for i in range(20):
    print(generate(gram = create_gramma(bot,split="="),target="bot")

您好我是5673号销售经理,请问你要想咨询车子哪方面的内容呢？
先生,您好我是2号销售经理,请问你要了解关于车子的哪些信息？
你好我是4号销售经理,请问你要询问哪一款车子？
你好我是2号销售经理,请问你要了解关于车子的哪些信息？
先生,您好我是7号销售经理,您需要了解关于车子的哪些信息？
女士,你好我是92号销售经理,请问你要想咨询车子哪方面的内容呢？
你好我是3633号销售经理,您需要询问哪一款车子？
你好我是13号销售经理,请问你要询问哪一款车子？
您好我是65号销售经理,请问你要了解关于车子的哪些信息？
您好我是21号销售经理,您需要想咨询车子哪方面的内容呢？
你好我是4号销售经理,请问你要想咨询车子哪方面的内容呢？
女士,您好我是29号销售经理,您需要询问哪一款车子？
你好我是8号销售经理,请问你要询问哪一款车子？
先生,你好我是75号销售经理,您需要了解关于车子的哪些信息？
先生,您好我是749317号销售经理,请问你要询问哪一款车子？
先生,你好我是981号销售经理,您需要想咨询车子哪方面的内容呢？
你好我是1号销售经理,请问你要想咨询车子哪方面的内容呢？
先生,您好我是43号销售经理,请问你要想咨询车子哪方面的内容呢？
先生,您好我是21号销售经理,您需要询问哪一款车子？
女士,你好我是2号销售经理,您需要询问哪一款车子？

  2. 使用新数据源完成语言模型的训练

  3. 获得最优质的语言
 
  
