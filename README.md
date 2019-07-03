# lesson-01 Assignment 
#1 复现课堂代码 

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
import random
random.choice(range(100））
filename="Result-1558435.csv"
import pandas as pd
content=pd.read_csv(filename,encoding="gb18030")
content.head()
article = content["content"].tolist()
len(articles)

import re
def token(string):
return re.findall("\w+,string)
from collections import Counter
with_jieba_cut =Counter(jieba.cut(articles[110]))
with_jieba_cut.most_common()[:10]
"".join(token(articles[110]))
articles_clean =["".join(token(str(a)) for a in articles]
len(article)
with open ("article_9k.txt","w") as f :
for a in articles_clean:
f.write(a+"\n"
def cut(string): return list(jieba.cut(string))
import jieba
def cut(string): retunr list(jieba.cut(string))
token=[]
for i , line in enmuerate((open("article_9k.txt"))):
  if i % 100 ==0: print (i)
  if i > 10000: break
  TOKEN+=cut(line)
from functools import reduce
fromoperator import Counter
words_count =Counter(TOKEN)
words_coun.most_common(100)
frequiences=[f for w , f in word_count.most_common(100)]
x = [i for i in range(100)]
%matplotlib inline 
import matplotlib.pylot as plt 
plt.plot(x,frequiences)
import numpy as np
plt.plot(x,np.log(frenquiences))
def prob_1(word):
prob_1()
TOKEN[:10]
TOKEN = [str(t) for t in TOKEN]
TOKEN_2_GRAM = ["".join(TOKEN[i:i+2]) for i in range(len(TOKEN[:-2]))]
TOKEN_2_GRAM[:10]
words_counts_2 = Counter(TOKEN_2_GRAM)
def prob_2(word):return words_count[word]/ len(TOKEN)
def prob_2(word1,word2):
  if word1 +  word2 in words_cont_2: return words_count_2[word1+word2] / len(TOKEN_2+GRAM)
  else:
  return 1 / len(TOKEN_2_GRAM)
  prob_2("我们”,"在”）
  prob_2("在“,"吃饭”）
  prob_2("去“，"吃饭"）
  
  def get_probablity(sentence):
  words = cut(sentence)
  sentence_pro = 1
  for i i, word in enumerate(words[:-1]:
  next_ words[i+1]
  probability = pron-2（word,next)
  sentence_prob * = probability
  return sentence_pro 
  
  get_probablity("小明今天抽奖抽到一台苹果手机")
  get_probablity("小明今天抽奖抽到一架波音飞机")
  get_probablity("洋葱奶昔来一杯")
  get_probablity("养乐多绿来一杯")
  
  for sen in [generate(gram = example_grammar,target ="sentence") for i in range(10)]:
  print("sentence:{} with Pro:{}".foramt(sen,get_probablity(sen))
  need_compared=[
  "今天晚上请你吃大餐，我们一起吃日料 明天晚上请你吃大餐，我们一起吃苹果",
    "真事一只好看的小猫 真是一只好看的小猫",
    "今晚我去吃火锅 今晚火锅去吃我",
    "洋葱奶昔来一杯 养乐多绿来一杯"]
    
    for s in need _compared:
    s1,s2 = s.split()
    p1,p2 = get_probablity(s1),get_probablity(s2)
    better = s1 if p1> p2 else s2 
    
    print("{} is more possible".fomat(better)) "]
    print("-"*4 +' {} with probablity {}.format(s1,p1))
    print("-"*4 {} with probablity {}.format(s2,p2)
    
    Data Driven

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
    
    
  
  
  

