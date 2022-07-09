# 起始目录

`*.py`

`paper.xlsx`

`institution_map.xlsx`

`译文_institution_map.xlsx`

(文件名暂时是写死的)

# 运行

菌群:修改options.py中
```python
dataset = bacterium
```

`python utils.py`

`python main.py`

疾病:修改options.py中
```python
dataset = disease
```

`python utils.py`

`python main.py`

# 结果

bacterium:
`
Dev P : 83.76%, Dev R : 75.97%, Dev F1 : 79.67%
Test P : 88.29%,Test R : 74.24%,Test F1 : 80.66%
`

disease:
`
Dev P : 94.48%, Dev R : 95.80%, Dev F1 : 95.14%
Test P : 94.48%,Test R : 95.14%,Test F1 : 94.81%
`
