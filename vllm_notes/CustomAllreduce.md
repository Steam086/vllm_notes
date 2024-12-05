### 符号解释

| name        | 说明                                                                                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| world_size  | node中的GPU数量                                                                                                                                                                                               |
| rank_data   | 一个分配在GPU上的指针池，其中C++中对RankData的定义是 ``struct{void * ptrs[8]}``这里取8是因为CustomAllreduce操作支持的最大GPU数量是8，这几个指针分别指向同一node上的多个GPU上的即将用于allreduce操作的输入变量 |
| "register"  | 下面函数的标识符中有register存在，register应该表示的是： 将内存中的RankData拷贝到GPU显存的rank_data池中                                                                                                       |
| handle和ptr | handle是通过CUDA进程间通信（IPC）函数获取的返回值，可以传递给其他进程并在其他进程通过OpenIpcHandle打开以获取ptr                                                                                               |

### API overview

- [create_shared_buffer](#create_shared_buffer)/free_shared_buffer ：	创建、释放共享内存（GPU上的内存）
- [capture](#capture): @contextmanager	The main responsibility of this context manager is the `register_graph_buffers` call at the end of the context.
- [register_graph_buffers](#register_graph_buffers)
- should_custom_ar	：判断是否应该使用custom_all_reduce
- [all_reduce](#description) :custom_all_reduce调用的函数，调用了cuda定义的函数
- [custom_all_reduce](#custom_all_reduce)	对外调用的接口

### Details

- #### create_shared_buffer

`<a id="create_shared_buffer"></a>`

| 输入参数 | 类型 | 默认值 | 说明 |
| -------- | ---- | ------ | ---- |
| void     |      |        |      |

| 返回字段 |   类型   | 说明                           |
| -------- | :-------: | ------------------------------ |
| ptrs     | List[int] | 一个指针数组，大小为world_size |

创建共享内存并返回指向共享内存的指针$ptrs$,其中调用了CUDA内存分配函数，并使用OpenIpcHandle打开了其他同一node上其他设备的共享内存handle。

- #### capture:

`<a id="capture"></a>`

| 输入参数 | 类型 | 默认值 | 说明 |
| -------- | ---- | ------ | ---- |
| void     |      |        |      |

| 返回字段 | 类型 | 说明 |
| -------- | :--: | ---- |
| void     |      |      |

这个函数是一个 `@contextmanager`，主要目的是在graph_capture最后调用 `register_graph_buffers`，将所有allreduce用到的输入地址注册到rank_data中。

解释：这个函数仅用于CUDA graph模式中，在CUDA graph 模式中，所有的操作不会立即被执行，CUDA会根据操作预先构建计算图，并一次性提交到GPU中执行，其中allreduce操作进行进程间通信需要将input注册到 `rank_data`中，这个注册的操作不会每次调用allreduce都执行一次，会在调用allreduce时将需要注册的ptr存入一个待注册数组（`graph_unreg_buffers_`）中，等到调用 `register_graph_buffers`时再将这些未被注册的ptr 进行 1. allgather获取其他进程中的handles。 2. 将这些获取到的handles打开并注册到 `rank_data`中

<!-- The main responsibility of this context manager is the `register_graph_buffers` call at the end of the context. It records all the buffer addresses used in the CUDA graph. -->

- #### all_reduce

`<a id="description"></a>`先进行一个条件的判断（是否处于CUDA graph 模式）如果不处于CUDA graph 模式，直接将input拷贝到预先分配的GPU buffer中，如果处于CUDA graph模式，直接input放入 `graph_unreg_buffers_`并进行allreduce操作。前面解释了[这样做的原因](#capture)

在C++函数内部有更细节的处理：

如果满足一些特定条件（full_nvlink_且输入Tensor比较大，在world_size<=4时的阈值为512KB，world_size<=8时的阈值为256KB），将调用 `cross_device_reduce_2stage`（CUDA核函数），否则调用 `cross_device_reduce_1stage`

`cross_device_reduce_2stage`详细解释：

- **stage 1: reduce scatter**
  首先，节点中的所有GPU只负责一部分的reduce，比如对于一个GPU的rank=rank，它负责处理 `input[start:end]` ，其中

  ```apache
  part = size / ngpus; //size 是输入张量的大小
  start = rank * part ; 
  end = rank == world_size - 1 ? size : start + part; 
  ```

  将这一部分reduce之后的结果放入一个预先分配的shared_memory中
- **stage 2: allgather.**

  每个GPU读取shared_memory 中的数据，并将这些数据copy到result（最终的返回结果)中。

重要代码简化版（部分同步代码省略）：

- 第一阶段

```apache
for (int idx = start + tid; idx < end; idx += stride) {
    // 将reduce结果存入保存中间结果的共享内存
    tmp_shared_buf[rank][idx] = packed_reduce(ptrs,idx);//
}
```

- 第2阶段：

```apache
// allgather操作
for (int idx = tid; idx < largest_part; idx += stride) {
    for (int i = 0; i < ngpus; i++) {
        int gather_from_rank = ((rank + i) % ngpus);
        if (gather_from_rank == ngpus - 1 || idx < part) {
            int dst_idx = gather_from_rank * part + idx;
            result[dst_idx] = tmp_shared_buf[i][idx];
        }
    }
}
```