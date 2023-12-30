
# Conclusion
1. AdamW 需要对非plm参数赋予较高lr
2. plm的选择影响较小，large能提高1% (DeBerta更好)
3. consine、linear影响较小

## only ce (concate utterances)
### DeBerta-base linear
| valid | 64.46 | 64.28 | 65.49 |  |  |  |  |  |
| test  | 64.12 | 64.39 | 65.10 |  |  |  |  |  |
### DeBerta-base cosine
| valid | 64.94 | 63.56 | 66.28 |  |  |  |  |  |
| test  | 63.40 | 62.59 | 62.78 |  |  |  |  |  |

### DeBerta-large
| valid | 65.46 | 65.48 | 66.53 | 64.75 | 66.20 | 66.45 |  |  |
| test  | 65.88 | 65.53 | 66.57 | 66.38 | 65.72 | 65.93 |  |  |

context添加label信息 + 分层 classifier
### DeBerta-base cosine
| valid | 66.41 | 66.60 | 66.58 | 66.36 | 66.26 |  |  |  |
| test  | 65.83 | 63.98 | 65.41 | 64.60 | 64.04 |  |  |  |
### DeBerta-large
| valid | 66.11 | 
| test  | 67.02 | 

### roberta-base AdamW


### roberta-large
| valid | 65.82 | 66.11 | 65.43 | 66.01 | 67.58 | 65.59 |  |  |
| test  | 65.12 | 64.30 | 64.52 | 66.18 | 64.25 | 66.05 |  |  |

### simcse-base
| valid | 64.94 | 63.56 |  |  |  |  |  |  |
| test  | 63.40 | 62.59 |  |  |  |  |  |  |
### simcse-large
| valid | 63.69 | 63.87 | 63.26 | 
| test  | 63.46 | 64.15 | 63.22 |