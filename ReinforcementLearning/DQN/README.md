DQN 全称： Deep Q-Network（深度 Q 网络） = Deep Neural Network + Q-learning

DQN 是 off-policy 的

# 为什么需要 DQN

## Q-learning 的缺陷

当状态空间非常大或为连续空间时，Q 表会面临维度灾难，无法存储和有效更新所有状态-动作对的 Q 值。

解决方案：用神经网络来估计 Q 值

$$
Q(s,a; \theta)
$$

通过函数去拟合，其中 $\theta$ 是神经网络的参数。通过训练使网络输出逼近真实 Q 值，从而具备泛化能力，能够处理高维、连续的状态空间。

`Q值`：全称为`状态-动作价值函数`。表示处于状态 s 并选择动作 a，从长远来看，这个选择有多好。 Q 值越大：在该状态下采取该动作的长期收益越大。 Q 值越低：该动作在该状态下的长期收益越差（甚至可能是负面的）。

# DQN 解决的问题（DQN 神经网络的输出）

DQN 的想法：给定当前状态的`obs(state)`，神经网络输出==每个动作==的价值（即 Q 值）。

```
Q(obs, action_0)
Q(obs, action_1)
...
Q(obs, action_n)
```

## 示例

- 假设游戏只有四个动作：`up`, `down`, `left`, `right`,
- 当前状态表示为：`state`
- 神经网络输入为`state`，输出为`Q(state, up)`, `Q(state, down)`, `Q(state, left)`, `Q(state, right)`

```
流程图如下：
        State
          │
          ▼
    ┌───────────┐
    │ Neural    │
    │ Network   |
    | (DQN)     │
    └───────────┘
          │
          ▼
 ┌─────────────────┐
 │ Q(上)            │
 │ Q(下)            │
 │ Q(左)            │
 │ Q(右)            │
 └─────────────────┘
```

- 如果网络输出为：

```
[1.2, 0.3, 2.1, 5.7]
```

- 则表示每个动作预计长期收益为：

```
上 = 1.2
下 = 0.3
左 = 2.1
右 = 5.7
```

- 然后,由
  $$
  arg \quad max \ Q(s,a)
  $$
  得，Agent 选择的动作为`right`

# DQN 的一次训练过程

```
        当前状态 S
             │
             ▼
       ┌──────────┐
       │ Q Network│
       └──────────┘
             │
             ▼
       得到 Q(S,A)
             │
             ▼
        选择一个动作 A
             │
             ▼
    Environment 环境交互
             │
       ┌─────┴─────┐
       ▼           ▼
     Reward       S'
       │           │
       └─────┬─────┘
             ▼
        计算 Target
             │
             ▼
       更新神经网络

注：此时训练还不稳定。因此DQN还有两个关键技术。Experience Replay（经验回放） Target Network 目标网络
```

其中：
TD-Target 为

$$
y = r + \gamma \max_{a'} Q(s', a')
$$

然后让网络预测的:

$$
Q(s, a)
$$

尽量接近:

$$
y
$$

所以损失函数可以写成:

$$
L = \big(y - Q(s, a)\big)^2
$$

然后反向传播更新 $\theta$

# Experience Replay: 经验回放

## 现有问题

在 Q-learning 中，每次交互后，会使用样本 $(s, a, r, s')$ 直接更新 Q 表。但使用神经网络函数近似时，会带来两个问题：

1. 数据相关性： RL 的连续采样之间高度相关。 神经网络假设训练的数据是独立同分布的，强相关性会导致梯度更新方向高度一致，使得训练不稳定甚至发散。
2. 非平稳目标：Q-learning 的目标值 $r + \gamma \max_{a'} Q(s', a'; \theta)$ 本身依赖于当前网络参数 $\theta$。每次更新后目标也在变，相当于“追着移动的靶子射击”，极易震荡。

> **核心思想：** 把“数据采集”和“模型训练”解耦。先收集数据存起来，再从中随机抽样进行训练，打破时序相关性，同时复用历史数据提高样本效率。

## DQN 做法

每次 Agent 与环境交互后，不立即丢掉经验，而是作为五元组存在经验池当中。

1. 经验池 Replay Buffer：一个固定容量的循环缓冲区，通常大小为 $10^5 \sim 10^6$。每条记录是一个转移元组

   $$
   e_t = (s_t, a_t, r_t, s_{t+1}, \text{done}_t)
   $$

   当池满时，新数据覆盖最旧的数据（先进先出）。

   ```
   ┌──────────────────────────┐
   │ Replay Buffer            │
   ├──────────────────────────┤
   │ (S1,A1,R1,S2)            │
   │ (S2,A2,R2,S3)            │
   │ (S3,A3,R3,S4)            │
   │ ...                      │
   └──────────────────────────┘
   ```

2. 数据采集阶段
   智能体与环境交互时，使用 $\epsilon$-greedy 策略选择动作，将产生的转移存入经验池。**此过程不进行任何梯度更新。**

3. 训练阶段：从经验池中均匀随机采样一个小批量，然后用这抽出来的经验训练神经网络。

## 简易代码框架

```
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transtion):
        self.buffer.append(transtion)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

```

## 经验回放的优点

| 好处               | 说明                                                               |
| :----------------- | :----------------------------------------------------------------- |
| **打破时序相关性** | 随机采样使小批量样本近似独立同分布，满足 SGD 的理论前提            |
| **提高样本效率**   | 每个转移被多次用于训练，避免“用完即弃”；尤其对稀疏奖励环境至关重要 |
| **平滑训练过程**   | 混合了不同时期、不同策略下收集的数据，避免模型过度拟合近期轨迹     |

## ⚠️ 局限性与改进方向

- **均匀采样的低效性：** 所有样本被等概率采样，但高 TD 误差的样本包含更多信息。→ **优先经验回放**按 TD 误差绝对值赋予采样概率。
- **过时数据问题：** 早期策略收集的样本可能与当前策略差异很大，导致偏差。→ 可通过重要性采样校正或缩短回放窗口缓解。
- **内存开销大：** 存储百万级图像帧占用大量内存。→ 可使用压缩存储或仅存储潜在特征。
- **不适合纯在线/安全敏感场景：** 回放旧数据意味着可能重新学习已被证明危险的行为。→ 安全 RL 中常改用在线方法或约束回放内容。

## 总结

经验回放的本质是将强化学习从“在线序列学习”转化为“离线批量监督学习”的形式，使深度学习工具能够有效应用于序贯决策问题。它与目标网络共同构成了原始 DQN 的两大基石，后续几乎所有基于值函数的深度 RL 算法都继承了这一机制或其变体。

# Target Network: 目标网络

目标网络 是 DQN 中用于解决训练不稳定问题的核心机制

## 为什么需要目标网络？

回顾 Q-learning 的更新公式：

$$
Q(s, a; \theta) \leftarrow Q(s, a; \theta) + \alpha \left[ r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right]
$$

在表格型 Q-learning 中这没有问题，但当用神经网络 $Q(s,a;\theta)$ 做函数近似时，**同一个参数 $\theta$ 同时出现在等式两边**：

- **左边**：当前网络的预测值（被更新的对象）
- **右边**：目标值 $y = r + \gamma \max_{a'} Q(s', a'; \theta)$（也依赖于 $\theta$）

```
θ 被更新
 ↓
Q 发生变化
 ↓
Target 发生变化
 ↓
Loss 发生变化
 ↓
θ 又被更新
 ↓
Target 又变化
```

这意味着每次梯度更新后，**目标本身也在移动**。相当于“自己追自己的影子”，导致：

1.  **高度相关的振荡：** 预测值和目标值强耦合，误差信号反复放大。
2.  **发散风险：** 小的过估计通过 $\max$ 操作被传播和累积，Q 值可能爆炸式增长。
3.  **非平稳训练：** 违背了监督学习中“固定标签”的基本假设，SGD 的理论收敛性不再成立。

## 目标网络核心思想：

将目标值（相当于监督学习的标签）的计算与当前网络的更新解耦。用一个**冻结的、定期同步的网络副本**来生成目标，使目标在短时间内保持“准静止”。

而在 DQN 中，标签是由模型自己生成的（自举），这就破坏了监督学习“固定标签”的基本前提。

## 工作机制

```
Online Network 训练网络
Target Network 目标网络
```

DQN 维护两套网络参数：

| 网络         | 符号       | 作用                        | 更新方式                                        |
| :----------- | :--------- | :-------------------------- | :---------------------------------------------- |
| **训练网络** | $\theta$   | 输出 Q 值估计，接受梯度更新 | 每步 SGD 更新                                   |
| **目标网络** | $\theta^-$ | 仅用于计算目标值 $y$        | 每隔 $C$ 步硬拷贝：$\theta^- \leftarrow \theta$ |

### 具体流程

1.  初始化时令 $\theta^- = \theta$。
2.  每个训练步：
    - 从经验池采样小批量 $\{(s_i, a_i, r_i, s'_i, done_i)\}$
    - 用**目标网络**计算固定目标：
      $$y_i = r_i + \gamma \max_{a'} Q(s'_i, a'; \theta^-)$$
    - 用**训练网络**计算预测值 $Q(s_i, a_i; \theta)$
    - 最小化损失 $\mathcal{L}(\theta) = \frac{1}{B}\sum_i (y_i - Q(s_i, a_i; \theta))^2$，只更新 $\theta$
3.  每隔 $C$ 步（如 $C=10000$），执行一次硬同步：$\theta^- \leftarrow \theta$

## 优点

- **打破自举循环：** 目标值在 $C$ 步内不变，将非平稳问题转化为一系列近似平稳的子问题。
- **抑制过估计传播：** $\max$ 操作的偏差不会立即反馈到自身，给了当前网络“冷静修正”的时间窗口。
- **平滑学习曲线：** 目标的阶梯式变化比连续漂移更容易优化，大幅降低发散概率。

## 后续改进：软更新

# DQN 骨架

```
                 Environment
                      │
                      │ S
                      ▼
               ┌─────────────┐
               │ Q Network   │
               │ θ           │
               └─────────────┘
                      │
                      │ Q values
                      ▼
                  Action
                      │
                      ▼
               Environment
                 │       │
                 │       │
                 ▼       ▼
              Reward     S'
                 │       │
                 └───┬───┘
                     ▼
              Replay Buffer
                     │
                     │ 随机采样
                     ▼
               ┌─────────────┐
               │ Mini Batch  │
               └─────────────┘
                     │
                     ▼
             Target Network
                  θ⁻
                     │
                     ▼
              TD Target y
                     │
                     ▼
             ┌─────────────┐
             │    Loss     │
             │(y-Q)²       │
             └─────────────┘
                     │
                     ▼
               Backprop
                     │
                     ▼
               更新 θ
```

# DQN 两大基石

- Experience Replay
- Target Network

# DQN

环境以`CartPole`为例

## QNetwork：输入：状态，输出：每个动作的 Q 值

### state

CartPole 的 obs 是 4 维：

$$
obs = [x,x˙,θ,θ˙]
$$

分别表示

```
小车位置
小车速度
杆子角度
杆子角速度
```

### Action

```
0 → 向左
1 → 向右
```

因此，

```
输入：4维 State

        ↓

   Neural Network

        ↓

输出：2个 Q value
```

### 代码实现

[Qnetwork.py](./src/Qnetwork.py)

注：DQN 最终输出的是动作价值 Q-value，而不是传统分类网络的概率。因此不需要 Softmax

## ReplayBuffer：经验回放池

[buffer.py](./src/buffer.py)
DQN 不直接用最新一步训练，而是先把经验存起来：

```
(s, a, r, s', done)
```

然后每次随机抽取一个小批量，进行训练
优点：

- 打破时序相关性
- 提高样本效率（旧经验可以重复利用）
- 平滑训练过程

### 为什么需要 ReplayBuffer

因为不希望网络：

```
看到一条经验
↓
训练一次
↓
看到下一条非常相似的经验
↓
训练一次
```

而是应该：

```
Agent
 ↓
不断产生经验
 ↓
Replay Buffer
 ↓
随机抽取 Batch
 ↓
训练
```

例如：

```
Replay Buffer

┌──────────────────────┐
│ experience 1         │
│ experience 2         │
│ experience 3         │
│ ...                  │
│ experience 100000    │
└──────────────────────┘
          ↓
     random sample
          ↓
┌──────────────────────┐
│ batch 1              │
│ batch 2              │
│ ...                  │
│ batch 64             │
└──────────────────────┘
```

这样可以打破时间相关性

## DQN 真正的核心：TD Target

对于一条经验：

$$
(s, a, r, s', done)
$$

我们需要计算：

$$
y
$$

也就是目标 Q value。

如果下一状态不是终止状态：

$$
y = r + \gamma \max_{a'} Q(s', a')
$$

如果已经结束：

$$
y = r
$$

合起来：

$$
y = r + \gamma(1-done)\max_{a'} Q(s', a')
$$

但是由于 DQN 具有两个网络为训练网络和目标网络，分别记为$Q(s, a; \theta)$ 和 $Q(s, a; \theta^-)$
因此 Target:

$$
y = r + \gamma(1-done)\max_{a'} Q(s', a'; \theta^-)
$$

而预测值:

$$
Q(s, a; \theta)
$$

## DQN 的 loss

现在我们有：

预测值：

$$
Q(s, a; \theta)
$$

Target:

$$
y = r + \gamma(1-done)\max_{a'} Q(s', a'; \theta^-)
$$

所以：

$$
L = \big(y - Q(s, a; \theta)\big)^2
$$

## 经典 DQN 框架

```
                    Environment
                         │
                         ▼
                       State
                         │
                         ▼
                    ε-greedy
                         │
                         ▼
                       Action
                         │
                         ▼
                    Environment
                    ↙         ↘
                 Reward      Next State
                    ↘         ↙
                         ▼
                    Replay Buffer
                         │
                         │ sample batch
                         ▼
                ┌─────────────────┐
                │   Mini Batch     │
                └─────────────────┘
                    │           │
                    ▼           ▼
             Online Network   Target Network
                    │           │
                    ▼           ▼
                 Q(s,a)      max Q(s',a')
                    │           │
                    │           ▼
                    │       TD Target
                    │           │
                    └─────┬─────┘
                          ▼
                         Loss
                          │
                          ▼
                     Backprop
                          │
                          ▼
                  更新 Online Network
                          │
                          │ 每 N 步
                          ▼
                  更新 Target Network
```

# 代码部分

- [Qnetwork.py](./src/Qnetwork.py): 基于 MLP 实现了一个 DQN 的神经网络，输入的维度为 obs 的维度，输出为 action 的维度
- [buffer.py](./src/buffer.py): 实现了一个最简单的经验回放池，用于存储和随机抽取经验
- [dqnagent.py](./src/dqnagent.py): 基于 2015 年 DeepMind 的论文[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)实现了一个 DQN 的 Agent，包括训练和测试
- [main.py](./src/main.py): 用于训练 DQN 的 Agent，并查看训练过程
- [dqn_sb3.py](./src/dqn_sb3.py): 基于 stable-baselines3 实现了一个 DQN 的 Agent，包括训练和测试
