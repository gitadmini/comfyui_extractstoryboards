# ExtractStoryboards

提取分镜

## 节点

### ExtractStoryboards

提取分镜关键帧及索引

#### 输入

- threshold:阈值 建议0.1。越小越严格。如果视频动作幅度不大，可以提高一些阈值，0.3-0.4之间，不超过0.5为宜。
- mergeInterFrames：合并孤帧。当关键帧所在的片段总帧数小于mergeInterFrames时，向左合并，即只保留最右侧的关键帧。（对于有转场动画的视频有一定的效果）

### Int Batch Size

获取整数集合的长度

#### 输入

- ints 一个整数的集合

#### 输出

- ints_size 输出是这个集合中整数的总数量

### Int Batch

获取整数集合的某个值

#### 输入

- ints 一个整数的集合
- int_index 对应的索引

#### 输出

- int_value 输出是这个索引上的值

