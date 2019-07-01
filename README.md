# lesson-01-20190630-
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
 
 
  
