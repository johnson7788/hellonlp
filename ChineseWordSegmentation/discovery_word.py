# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:16:21 2020

@author: cm
"""


from ChineseWordSegmentation.utils import load_txt
from ChineseWordSegmentation.segment_entropy import get_words
from ChineseWordSegmentation.hyperparameters import Hyperparamters as hp
from ChineseWordSegmentation.utils import load_excel_only_first_sheet,ToolWord


vocabulary_set = set(load_txt(hp.file_vocabulary))


def get_words_new(corpus):
    """
    获取不在字典里面的新词
    """
    words = get_words(corpus)
    print('Length of words:',len(words))
    words_clean = [l for l in words if ToolWord().remove_word_special(l)!='']
    print('Length of words(clean):',len(words_clean))
    return [w for w in words_clean if w not in vocabulary_set]


if __name__ == '__main__':
    # document = ['葫芦屏 武汉 武汉 北京 葫芦屏 葫芦屏 武汉','武汉 武汉 北京 葫芦屏 葫芦屏 武汉 十四是十四四十是四十，']
    # ws = get_words(document)
    # print(ws)
    f = 'data/data.xlsx'
    contents = load_excel_only_first_sheet(f).fillna('')['content'].tolist()#[:5000]
    contents = contents[:30]
    print(f'句子个数： {len(contents)}')
    nws = get_words_new(contents)  
    print(nws[:200])
    print(len(nws))














