# ExtractStoryboards

提取分镜

使用方法请参考 <https://www.bilibili.com/video/BV1qBTvzfE5M/?vd_source=52e0e85c23c34bb5ace2c8ebc915dbaf>

## 历史

- v1.0.4 bug修复，对最后一个分镜增加了最小值和最大值限制的判断
- v1.0.3 ExtractStoryboards添加了参数maxFrames，可用来拆分长镜头

## 节点

### ExtractStoryboards

提取分镜关键帧及索引

#### 输入

- threshold:阈值 建议0.1。越小越宽松。如果视频动作幅度不大，可以提高一些阈值，0.3-0.4之间，不超过0.5为宜。
- mergeInterFrames：合并孤帧。当关键帧所在的片段总帧数小于mergeInterFrames时，向左合并，即只保留最右侧的关键帧。（对于有转场动画的视频有一定的效果）
- maxFrames:拆分长镜头。当分镜过长时，拆分成每个分镜帧数小于等于maxFrames。（对于接下来用wan2.1来处理的分镜，可以设置为81）

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

