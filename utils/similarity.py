import editdistance


def similarity_by_editdistance(term, tokens):
    """
    按编辑距离从 tokens 中找到与 term 最相似的词
    """
    edit_dis = []
    for token in tokens: 
        edit_dis.append(editdistance.eval(term, token))
    rank = sorted(
        range(len(edit_dis)), 
        key=lambda k: edit_dis[k], 
        reverse=False
        )
    return [tokens[idx] for idx in rank]

    words, new_words = term.split(' '), []
    for word in words:
        edit_dis = []
        for token in tokens: 
            edit_dis.append(editdistance.eval(word, token))
        idx = edit_dis.index(min(edit_dis))
        new_words.append(tokens[idx])
    
    return ' '.join(new_words) 

