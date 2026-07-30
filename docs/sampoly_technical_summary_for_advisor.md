# 基于SAMPoly的遥感建筑物交互式轮廓提取系统设计与实现

## 一、研究背景与目标

遥感建筑物提取是高分辨率遥感解译中的重要任务，其结果通常直接服务于城市建模、变化检测、地理信息更新与空间规划等应用。近年来，基于深度学习的自动分割方法已经能够较快地给出建筑物候选区域，但在复杂真实场景中，仍然容易受到以下因素影响：

- 建筑物之间粘连；
- 树木遮挡与阴影干扰；
- 屋顶纹理复杂、材质变化明显；
- 道路、广场等非建筑区域与建筑边界混淆；
- 自动结果边界不规则，难以直接作为高质量矢量结果使用。

因此，本项目并不单纯追求“完全自动”的建筑提取，而是尝试在公开数据集与现有模型基础上，构建一个**自动候选检测 + 人机交互精修**的建筑物交互式轮廓提取系统。

本项目的核心目标是：

1. 利用 SAMPoly 的 Auto 模式快速提供建筑候选区域；
2. 利用 Prompt 模式完成单建筑初始轮廓提取；
3. 通过 positive / negative point 进行多轮交互优化；
4. 实现面向大影像场景的逐 tile 浏览、编辑与保存；
5. 最终输出结构化、可扩展的建筑 polygon 结果。

从研究角度看，本项目关注的重点不是“是否能自动分出建筑”，而是**如何构建一个精度更高、可解释、可修正、适用于遥感实际应用的人机协同轮廓提取流程**。

---

## 二、总体技术路线

### 2.1 技术路线图（文字版）

```text
大范围遥感影像
    ↓
Stage 1：Tile切片与坐标映射建立
    ↓
Stage 2：Auto自动候选检测
    ↓
生成建筑候选 polygon / bbox
    ↓
Stage 3：进入Tile Review界面
    ↓
用户点击目标建筑
    ↓
根据点击位置自动匹配Auto bbox
    ↓
Stage 4：Prompt单建筑初始提取
    ↓
Stage 5：Positive / Negative Point多轮交互优化
    ↓
实时更新建筑轮廓
    ↓
结果质量控制与异常结果拒绝
    ↓
tile级编辑结果保存
    ↓
后续多tile融合与全局矢量输出
```

### 2.2 系统架构图（文字版）

```text
输入层
- 大幅遥感影像（GeoTIFF / TIFF / PNG / JPG）

↓

数据处理层
- 大影像切片
- Manifest生成
- Tile与原图坐标映射

↓

自动候选层
- SAMPoly Auto推理
- 候选polygon与bbox生成
- auto_results存储

↓

交互提取层
- Tile浏览
- 建筑点击
- Auto bbox匹配
- Prompt初始轮廓提取
- Positive / Negative Point交互优化

↓

结果约束层
- Mask后处理
- 主组件筛选
- Polygon提取
- Polygon质量检查
- Invalid refine rejection

↓

结果管理层
- edited_results保存
- TileResult统一协议
- 后续final_results / GIS导出扩展
```

---

## 三、系统设计

## 3.1 数据处理模块

在大范围遥感影像场景下，直接对整幅影像进行交互式分割会带来显著的显存压力、IO 压力和界面交互负担。因此，本项目首先构建了大影像切片工作流。

该模块完成的工作包括：

- 将大影像按固定大小切分为多个 tile；
- 支持 overlap，降低切片边界对建筑分割的影响；
- 为每个 tile 记录其在原始影像中的位置偏移；
- 生成标准 manifest 文件，供后续 Auto 推理与交互界面统一读取。

Manifest 中记录的核心信息包括：

- 原始影像名称；
- 原始影像宽高；
- tile_size；
- overlap；
- 每个 tile 的 `tile_id`；
- 每个 tile 的相对路径；
- `x_offset / y_offset`；
- 实际 `width / height`；
- 是否为边缘 tile。

因此，Tile 模块不仅承担了切图功能，更重要的是建立了**tile局部坐标 ↔ 原图全局坐标**之间的稳定映射关系，为后续逐 tile 处理与全局融合奠定了基础。

---

## 3.2 Auto检测模块

### 3.2.1 模块目标

Auto 模块的主要目标并不是输出最终高精度建筑边界，而是：

1. 提供建筑候选位置；
2. 提供后续 Prompt 所需的初始 bbox；
3. 降低人工在大影像中搜索目标建筑的成本。

换句话说，Auto 阶段承担的是“候选检测”和“交互入口定位”的角色，而不是最终矢量结果生成。

### 3.2.2 当前实现方式

当前 Auto 模块分为两层：

- 原始模型推理入口：`infer_auto.py`
- tile工作流批量入口：`infer_tiles_auto.py` 与 `workflow/auto_infer_adapter.py`

其中，`infer_auto.py` 负责：

- 读取 Auto 配置；
- 加载 checkpoint；
- 构建预测 dataloader；
- 运行原始 SAMPoly Auto 推理；
- 输出 `results.json` 与 `results_mask.json`。

README 中给出的典型配置为：

- config：`configs/auto_whumix.py`
- checkpoint：`checkpoints/auto_whumix.pth`

### 3.2.3 Auto输出数据格式

当前二次开发工作流并不直接使用原始 `results.json` 作为界面输入，而是将其转换为统一的 TileResult 结构。每个 tile 会生成一个单独的 JSON 文件，内部每个建筑 polygon 至少包含：

- `id`
- `tile_id`
- `status=active`
- `source=auto`
- `score`
- `points_tile`
- `bbox_tile`
- `points_global`
- `click_history`
- `metadata`

其中 `metadata` 中还会保留原始 Auto 结果中的信息，例如：

- `raw_index`
- `score_cls`
- `category_id`
- `area`

这意味着 Auto 阶段的输出已经从“模型推理结果”转换为“工程可用的交互基础数据结构”。

### 3.2.4 Auto结果如何被Prompt阶段调用

Auto 结果在当前系统中主要通过以下方式被后续 Prompt 阶段使用：

1. 在 tile review 界面加载当前 tile 的 `auto_results/tile_xxxx.json`；
2. 用户点击某个建筑位置；
3. 程序根据点击点，在当前 tile 的 Auto polygons 中自动选择一个最合适的 bbox；
4. 将该 bbox 作为 Prompt 单建筑提取的初始约束；
5. 后续所有 refine 都围绕该单建筑 bbox 与交互点进行。

当前 bbox 选择策略为：

- 优先选择**包含点击点**的 Auto bbox；
- 如果没有 bbox 包含点击点，则选择距离点击点最近、且在阈值范围内的 bbox；
- 若 Auto 候选不足，则退回 fixed bbox。

因此，Auto 模块为 Prompt 提供的是“候选位置 + 初始局部搜索范围”，显著降低了用户从整幅 tile 中手工定位建筑的成本。

---

## 3.3 Prompt交互模块

Prompt 模块负责将用户点击行为转化为单建筑轮廓提取结果，是系统从“自动检测”走向“可控精修”的关键桥梁。

### 3.3.1 用户点击如何触发Prompt

在 tile review 界面中：

- 用户左键点击目标建筑；
- 系统读取点击点在图像坐标系中的位置；
- 如果当前没有运行中的 Prompt Session，则创建新的 `PromptReviewSession`；
- 根据点击点自动匹配 Auto bbox；
- 将首次点击同时视为一个 positive point；
- 调用 Prompt segmentation 模型，生成该建筑的初始 preview polygon。

这意味着系统已经从“手动画 bbox”转变为“单点触发单建筑提取”，交互成本显著降低。

### 3.3.2 bbox如何匹配

当前 bbox 匹配逻辑的本质是“点击驱动的 Auto bbox provider”：

- 从 Auto polygons 中提取每个候选的 bbox；
- 计算点击点是否位于 bbox 内；
- 若有多个候选包含点击点，则优先选择得分更高的候选；
- 若没有包含点击点的候选，则根据点击点到 bbox 的距离选择最近候选；
- 若最近距离也超过阈值，则退回 fixed bbox。

这样做的意义在于：

- 保留 Auto 模型提供的位置先验；
- 同时避免用户必须精确点击到边界内部；
- 提升 Prompt 初始轮廓的稳定性。

### 3.3.3 point如何保存

当前系统对交互点的保存分为两套结构：

1. 语义集合：
   
   - `positive_points`
   - `negative_points`

2. 顺序集合：
   
   - `click_order`
   - `prompt_points`
   - `prompt_labels`

其中：

- `positive_points / negative_points` 用于按语义分类统计；
- `click_order` 记录真实用户点击顺序；
- `prompt_points / prompt_labels` 则是送入模型的有序点序列。

这样做的原因是，Prompt refine 不只是“加了哪些点”，而是“用户按什么顺序加了这些点”，因此需要保留点击时序信息。

### 3.3.4 positive / negative point如何传递给模型

在 Prompt Adapter 中：

- positive point 会编码为 label=1；
- negative point 会编码为 label=0；
- bbox 中心点通常也会作为辅助正提示；
- 所有点在 crop 空间内转换后传入 predictor；
- session 复用图像 crop embedding，避免每次迭代都重新编码整图。

因此，当前系统已经实现了真正意义上的**多轮 Prompt 交互会话**，而不是每次点击都从头运行一次独立推理。

### 3.3.5 mask如何生成

Prompt 推理后，模型首先生成 crop 空间中的 mask。随后系统会根据配置选择是否进行 mask 后处理，包括：

- connected component refine
- distance refine
- negative feature refine

这些后处理不改变模型结构，而是在模型输出 mask 的基础上做规则级修正，以增强交互修正效果。

### 3.3.6 mask如何转换为polygon

当前的 mask→polygon 流程包括：

1. 在 crop 空间中得到 refined mask；
2. 提取外轮廓；
3. 选择主轮廓（优先含 positive point 的轮廓，否则取最大轮廓）；
4. 使用 `approxPolyDP` 做适度简化；
5. 将 polygon 从 crop 坐标映射回 original tile 坐标；
6. 输出为 `points_tile`；
7. 再交由后续 polygon 质量检查决定是否真正接受。

因此，当前系统已经不只是“模型出什么就收什么”，而是开始建立一个**带工程约束的交互式 polygon 生成流程**。

---

## 3.4 Refine Session设计

为了支持多轮交互优化，系统设计了显式的 Refine Session 机制。其核心思想是：**每次 refine 只更新 preview，不直接覆盖最终结果；只有用户确认后才真正提交。**

### 3.4.1 当前session状态管理

Session 当前包含以下生命周期状态：

- `idle`
- `running`
- `accepted`
- `cancelled`

无论是 Prompt-first Session 还是对已有 polygon 的 Refine Session，都会维护自己的状态机。

### 3.4.2 初始polygon保存

Refine Session 会在启动时保存：

- `original_polygon`
- `current_polygon`

其中：

- `original_polygon` 用于取消时恢复；
- `current_polygon` 用于界面中实时显示 cyan preview。

### 3.4.3 每次迭代结果如何更新

每次用户添加一个 positive 或 negative point 后，系统会：

1. 复制当前点集合；
2. 增加新的点；
3. 调用 Prompt 推理；
4. 生成新的 candidate polygon；
5. 执行 refined polygon 质量检查；
6. 如果通过，则更新 `current_polygon`；
7. 如果失败，则拒绝更新，保留上一版 preview。

因此，当前系统已经支持“异常 refine 不覆盖上一版有效结果”的基本保护机制。

### 3.4.4 当前版本管理机制

当前版本管理主要依赖以下字段：

- `history`
- `current_polygon`
- `last_refine_validation`

其中：

- `history` 记录被接受的 preview 版本；
- `current_polygon` 表示当前有效 preview；
- `last_refine_validation` 记录最近一次 refine 的校验结果与 reject reason。

虽然还没有完整实现 undo / 最佳版本自动回退等高级功能，但版本管理基础已经具备。

---

## 3.5 可视化交互模块

当前系统已经实现了轻量级 tile review 界面，并支持较完整的交互操作。

### 3.5.1 当前已实现的界面能力

- tile 图像显示；
- auto polygons 加载与可选显示；
- edited_results 优先加载；
- Prompt preview polygon（cyan）显示；
- 选中 polygon 高亮；
- positive / negative point 可视化；
- 快捷键驱动的浏览、保存、接受、取消操作。

### 3.5.2 当前支持的交互操作

目前已经支持：

- tile切换；
- 建筑选择；
- 左键触发单建筑 Prompt 提取；
- positive point 添加；
- negative point 添加；
- 多轮 refine；
- 结果实时更新；
- Delete / Restore；
- 结果保存；
- 会话取消与接受。

也就是说，从“浏览 tile”到“完成建筑轮廓人工优化并保存”，当前流程已经是完整闭环。

---

## 四、现有交互优化的实现逻辑（重点）

这一部分是当前系统最核心的创新点。

## 4.0 交互优化处理机制详解

当前系统中的交互优化，并不是“用户点击一个正样本点或负样本点后，程序直接在 polygon 顶点层面做几何修改”，而是采用一种更接近**条件分割 + 规则后处理 + 结果质量控制**的联合机制。

换句话说，用户的点击不是直接改边，而是作为新的 Prompt 条件重新驱动模型推理。系统再基于模型输出的 mask 做局部 refine，最后再将 refined mask 转换成 polygon，并判断本轮结果是否足够可靠。

整体流程如下：

```text
用户点击 positive / negative point
    ↓
系统把点击点加入当前 session
    ↓
将 bbox + 历史点击点 + 当前点击点 一起送入 Prompt 模型
    ↓
模型重新输出新的 mask
    ↓
对 mask 做 refine 后处理
    ↓
从 refined mask 中提取主轮廓 polygon
    ↓
对 polygon 做质量检查
    ↓
若结果有效，则更新 preview polygon
若结果无效，则拒绝更新，保留上一版轮廓
```

因此，当前交互优化的本质可以概括为：

> 用户点击点 → 重新驱动模型预测 → 对模型输出 mask 做后处理 → 将 mask 转为 polygon → 决定是否接受本轮 refine

### 4.0.1 用户点击 positive point 后，模型是如何运行的

当用户点击一个 positive point 后，系统会将这个点加入当前交互 session，并将其编码为 Prompt 模型的前景提示，即 `label=1`。

其作用可以分为两层：

#### （1）模型层作用

positive point 的含义是：

> 该点附近区域属于目标建筑，应当被保留。

因此，在下一轮 Prompt 推理中，模型会结合：

- 当前 bbox；
- 历史 positive / negative point；
- 当前新加入的 positive point；

共同重新预测新的目标 mask。

这意味着 positive point 不是简单在已有轮廓中“打一个保留标记”，而是通过 Prompt 机制重新影响模型对建筑前景区域的判断。

#### （2）后处理层作用

positive point 同时还会参与 refine 后处理逻辑。例如在多个连通域同时存在时，系统会优先保留包含 positive point 的区域，以避免目标建筑主体被错误删除。

因此，positive point 的作用不是单一的，而是同时参与：

- 模型重预测；
- 主组件筛选；
- 结果稳定性保护。

### 4.0.2 用户点击 negative point 后，模型是如何运行的

negative point 的目标是告诉系统：

> 这个位置不应属于目标建筑，应当被从当前轮廓中删除。

在实现上，negative point 会被编码为 `label=0`，并与正样本点一起送入 Prompt 模型。模型会据此重新输出新的 mask。

但当前系统并不是“模型输出后就直接接受”，而是在模型输出之后继续做进一步的 refine。也就是说，negative point 会同时作用在：

1. Prompt 模型输入层；
2. mask refine 后处理层。

这是当前系统交互优化机制最重要的特点之一。

### 4.0.3 当前 negative refine 是如何进行轮廓优化的

当前 negative refine 并不是简单“根据负点找同类区域然后直接删除”，而是通过几种不同的后处理方式，对模型已经给出的 mask 再做局部修正。

#### （1）Connected Component Refine

这是一种相对保守、稳定的方式。

基本思想是：

- 先分析当前 mask 中有哪些连通域；
- 检查 positive point 落在哪些连通域中；
- 检查 negative point 落在哪些连通域中；
- 在这些连通域中筛选出最有可能属于目标建筑主体的区域。

当前工程实现进一步增强为：

- 如果多个连通域中有 positive point，则优先保留**包含 positive point 的最大面积组件**；
- 如果没有 positive point 命中组件，则保留最大的前景组件；
- 与目标主体无关的碎片全部删除。

因此，connected component refine 更接近一种：

> 基于正负点约束的区域级筛选

而不是逐像素做粗暴删除。

#### （2）Distance Refine

当错误区域和目标建筑已经粘连成同一个连通域时，仅靠 connected component 往往分不开，这时系统可以使用 distance refine。

其基本思想是：

- 对当前 mask 内每个前景像素，分别计算它到 positive point 和 negative point 的距离；
- 如果某个像素更接近 negative point，则倾向于将其移除；
- 如果某个像素更接近 positive point，则倾向于将其保留。

从本质上讲，distance refine 相当于：

> 在已有 mask 内部重新划分“更像目标主体”和“更像错误区域”的部分

这种方式比 connected component 更灵活，但也更容易把 mask 切碎，因此必须配合后续主组件保留与 polygon 质量检查使用。

#### （3）Negative Feature Refine

这是当前系统中更偏实验性的增强机制，用于处理“空间上相连，但视觉特征明显不同”的情况。

其逻辑是：

- 取 negative point 周围的小图像块作为负样本区域；
- 提取其颜色 / 纹理统计特征；
- 在当前 mask 内查找与该负样本区域视觉上相似的像素；
- 将这些相似区域从 mask 中进一步抑制或删除；
- 同时保护 positive point 附近区域，避免误伤目标主体。

这一机制比 distance refine 更接近：

> 根据负点样本识别“视觉上相似的同类区域”，再将其从当前 mask 中去除

但需要说明的是，当前版本识别的是**视觉相似区域**，而不是严格语义层面的目标类别判断。

### 4.0.4 为什么模型输出后还需要 mask 后处理

如果只依赖模型的单次输出，多轮 positive / negative point refine 后容易出现以下问题：

- mask 被切成多个碎片；
- 建筑主体和背景区域同时残留；
- 边界局部产生锯齿、毛刺或断裂；
- 极端情况下 polygon 会退化成杂乱线集合。

因此，当前系统不是“模型输出什么就直接用什么”，而是增加了额外的后处理与约束步骤：

1. **主组件保留**  
   保证只有单个主建筑区域继续进入 polygon 提取；

2. **polygon 质量检查**  
   如果 refine 后的 polygon 在几何上异常，则拒绝覆盖上一版结果。

### 4.0.5 mask 是如何转换为 polygon 的

在当前实现中，mask 转 polygon 的流程并不是简单取边界点，而是带有明确筛选策略：

1. 模型输出 crop 空间下的二值 mask；
2. refine 后只保留单个主组件；
3. 提取外轮廓；
4. 优先选择包含 positive point 的轮廓；
5. 如果没有 positive point 命中轮廓，则选择最大轮廓；
6. 使用 `approxPolyDP` 对轮廓进行适度简化；
7. 将 polygon 从 crop 坐标映射回原始 tile 坐标；
8. 交由 polygon 质量检查模块决定是否接受。

因此，当前 polygon 不是由点击点直接“连线生成”的，而是：

> 点 → 驱动模型重新预测 mask → refined mask → 主轮廓 → polygon

### 4.0.6 为什么要增加 invalid result rejection

在多轮交互优化中，最大的风险不在于“某一次模型输出不够理想”，而在于：

> 某一次异常 refine 结果直接覆盖了上一版本来正确的轮廓。

为避免这种情况，当前系统已经加入 invalid result rejection 机制。每次新的 candidate polygon 生成后，系统都会检查：

- polygon 点数是否异常；
- polygon 面积是否过小；
- 相比上一版面积是否发生突变；
- bbox 与当前 session 的目标区域是否仍然高度重叠；
- polygon 是否存在明显自相交；
- polygon 是否包含 NaN / Inf；
- refine 后结果是否由异常碎片导致。

如果本轮结果不满足条件，则：

- 不更新 `current_polygon`；
- 不增加有效 preview version；
- 不覆盖上一版轮廓；
- 控制台输出 reject reason。

也就是说，当前系统已经从：

> 每轮 refine 都直接更新结果

升级为：

> 每轮 refine 先生成候选结果，再通过质量门槛决定是否接受

### 4.0.7 一句话概括当前交互优化机制

如果在汇报中需要一句话概括当前系统的交互优化处理逻辑，可以表述为：

> 当前系统不是直接在 polygon 上编辑顶点，而是把用户的 positive / negative point 作为新的 Prompt 条件，重新驱动模型预测建筑 mask；随后结合 connected component、距离约束以及负样本特征抑制对 mask 做局部优化，再从优化后的单主组件 mask 中提取 polygon，并通过几何质量检查决定是否接受本轮 refine。

## 4.1 Prompt-first交互逻辑

当前系统采用 Prompt-first Review 模式，即：

- 用户不是先选择已有 Auto polygon 再改；
- 而是直接在 tile 上点击目标建筑；
- 系统自动匹配 bbox；
- 自动触发 Prompt；
- 输出该建筑的初始轮廓。

这样可以把 Auto 结果从“最终答案”降级为“候选位置提供者”，从而更符合交互式单建筑提取的目标。

## 4.2 Positive point优化逻辑

positive point 的作用是：

- 强化目标建筑应被保留的区域；
- 在多轮 refine 中为系统提供“这部分仍然是目标”的约束。

在实现上，positive point 会：

- 保存到 `positive_points`；
- 进入 `click_order`；
- 转换为 `prompt_labels=1`；
- 与 bbox 一起传入模型；
- 在 point refine / feature refine 中也起到主组件选择与区域保护作用。

## 4.3 Negative point优化逻辑

negative point 的目标是删除错误区域，例如：

- 与目标建筑粘连的相邻建筑；
- 屋顶外扩到道路或空地的区域；
- 阴影或树木覆盖导致的误分部分。

当前已经实现的核心逻辑包括：

### （1）负点有效性判定

为了避免“无关负点扰动当前轮廓”，系统在加入 negative point 前会判断该点是否有效。

当前默认规则是：

- 以**当前用户看到的 cyan preview polygon**为准；
- 使用 `pointPolygonTest` 判断点是否在 polygon 内部或边界附近；
- 支持一定像素容差；
- 若点位于 preview 外部，则忽略该负点，不加入 session，不触发 Prompt。

这解决了早期出现的“外部负点也导致轮廓变化”的问题。

### （2）mask后处理

negative point 进入模型后，系统会根据设置执行 mask refine：

- connected component refine
- distance refine
- negative feature refine

其目的不是改变模型，而是在 mask 层面对错误区域进行局部删除。

### （3）主组件保留

为避免多轮 negative refine 把 mask 切成多个碎片，当前系统新增了“主组件保留”逻辑：

- 如果多个组件中有 positive point，则保留含 positive point 的最大组件；
- 如果没有 positive point 命中组件，则保留最大前景组件；
- 其余碎片全部删除。

这一步的作用是：**即使 refine 后 mask 被切碎，也只允许单个主建筑区域继续进入 polygon 提取。**

### （4）polygon质量控制

即便 refine 后只保留了一个主组件，生成的 polygon 仍然可能异常，因此当前系统又增加了一层 quality gate。

当前至少会检查：

- polygon 点数是否过少或过多；
- polygon 面积是否过小；
- 与上一版面积相比是否突变；
- bbox 与 session bbox 是否仍有较高重叠；
- 是否包含 NaN / Inf；
- 是否明显自相交。

若校验失败，则：

- 拒绝本次 refine；
- 保留上一版 `current_polygon`；
- 不增加有效 preview version；
- 控制台输出 reject reason。

这意味着系统已经从“多轮 refine 直接覆盖结果”升级为“多轮 refine 受质量门槛约束的安全更新”。

## 4.4 结果保存逻辑

当前系统采用“preview 与 committed result 分离”的保存设计：

- session 中的 `current_polygon` 只是内存中的实时预览；
- 用户按 Enter 接受后，才将当前 polygon 写入 `edited_results`；
- 每个 tile 独立保存为统一 TileResult JSON；
- 再次进入该 tile 时，edited 结果优先加载。

这一机制有两个重要价值：

1. 用户可以安全地多轮试验 refine，而不会立刻破坏已有结果；
2. 系统具备了断点续做与可追踪编辑历史的工程基础。

---

## 五、当前实现成果

截至目前，系统已经完成以下关键能力：

- 大影像切片；
- Tile Manifest 生成；
- tile浏览；
- Auto批量推理；
- Auto结果统一格式转换；
- auto结果加载；
- 单建筑点击提取；
- bbox自动匹配；
- Prompt单建筑初始轮廓提取；
- positive point优化；
- negative point优化；
- polygon实时更新；
- invalid negative point过滤；
- refine后mask主组件保留；
- invalid refine rejection；
- tile级编辑结果保存。

从系统视角看，当前已经形成了“从大影像到单建筑交互精修”的完整原型闭环。

---

## 六、当前问题与改进方向

当前主要问题已经不只是模型本身，而更多集中在交互优化策略和结果约束机制上。

### 6.1 Auto结果边界不规则

Auto阶段结果仍然存在局部凸起、边界粗糙等现象。但这在设计上是可接受的，因为 Auto 的职责是候选检测，而不是最终轮廓精修。

### 6.2 Prompt首次提取可能包含非建筑区域

原因主要包括：

- 初始 bbox 中可能包含相邻建筑或非建筑区域；
- 单次点击信息量有限；
- 模型在局部区域内会对相邻高响应区域产生联动。

因此，Prompt 第一次输出更适合作为“交互起点”，而不是最终结果。

### 6.3 Negative point refine仍是当前最脆弱的部分

从代码与测试现象看，Negative refine 的主要风险来源于：

1. mask后处理可能把建筑区域切碎；
2. connected component 选择不稳定；
3. 多碎片 mask 直接进入 polygon extraction；
4. polygon extraction 后缺少质量门槛；
5. invalid refine 若不拒绝，就会覆盖正确 preview。

也就是说，当前系统面对的核心问题已经从“能否输出建筑轮廓”转向“如何在多轮交互中稳定、可靠地保住正确结果”。

---

## 七、下一步工作计划

下一阶段建议重点围绕以下五个方向推进：

### 7.1 完善negative point refine

- 进一步增强负点后处理的稳定性；
- 优化多负点累积作用机制；
- 减少对目标建筑主区域的误伤。

### 7.2 增加polygon质量评价

- 加强多边形几何合理性检查；
- 增加边长分布、形状极端性等约束；
- 进一步降低异常 polygon 覆盖有效结果的概率。

### 7.3 增加自动保存最佳结果

- 不只是保存最后一次结果；
- 而是保存当前会话中质量最稳定、最合理的结果；
- 为后续 undo / best-preview 机制打基础。

### 7.4 实现多个tile结果融合

- 处理跨 tile 边界建筑；
- 解决重复结果与边界断裂问题；
- 逐步从 tile 级编辑走向全图级矢量输出。

### 7.5 输出完整建筑矢量数据

- 生成全局 final_results；
- 支持 GIS / 矢量格式导出；
- 为后续 QGIS 插件集成与规则化处理提供基础。

---

## 八、汇报总结

本项目当前已经完成了从大影像切片、Auto候选检测、Prompt单建筑提取，到 positive / negative point 交互优化、preview质量控制与结果保存的完整原型流程。

从科研视角来看，当前阶段的主要成果不只是实现了交互式建筑提取界面，更重要的是建立了一个相对清晰的人机协同轮廓提取框架：

- Auto 负责候选定位；
- Prompt 负责单建筑初始提取；
- positive / negative point 负责局部交互修正；
- session 负责版本管理；
- polygon validation 负责结果约束。

当前系统的主要问题已逐渐从“模型分不分得出来”转向“多轮交互后如何稳定约束 refine 结果”，这也意味着后续工作的重点将逐步集中在交互优化策略、结果鲁棒性与全局融合机制上。

---

## 九、3–5分钟口头汇报稿

老师好，我汇报一下当前基于 SAMPoly 的遥感建筑物交互式轮廓提取系统的设计与实现进展。

本项目的目标不是单纯做一个全自动建筑提取模型，而是构建一个“自动候选检测 + 人机交互精修”的系统。原因在于，在遥感场景中，完全自动方法虽然能够快速给出建筑候选区域，但在复杂区域仍然容易受到阴影、树木、相邻建筑粘连等因素影响，导致边界不规则、误分和漏分。因此，我们希望把模型的自动能力和人的交互判断结合起来，在保证效率的同时提高结果质量与可控性。

当前系统整体可以分为四个阶段。第一阶段是大影像切片，我们将大范围遥感影像切成固定大小的 tile，并保存 tile 与原图之间的坐标映射。第二阶段是 Auto 自动候选检测，我们对所有 tile 运行 SAMPoly Auto 模式，得到建筑候选 polygon 和 bbox。这里 Auto 的作用主要是提供候选位置和 bbox，而不是直接作为最终结果。第三阶段是 Prompt 单建筑提取，用户在 tile review 界面中点击目标建筑，系统根据点击位置自动匹配 Auto 阶段提供的 bbox，然后调用 Prompt segmentation 模型，得到该建筑的初始轮廓。第四阶段是正负样本点交互优化，用户通过添加 positive point 保留目标建筑区域，通过 negative point 删除错误区域，系统多轮调用 Prompt refine，并实时更新当前预览轮廓。

在实现层面，目前已经完成了大影像切片、tile 浏览、Auto 批量推理、auto 结果加载、单建筑点击提取、bbox 自动匹配、Prompt 初始轮廓生成、positive / negative point 迭代优化，以及 tile 级编辑结果保存。系统已经形成了从大影像到单建筑交互式精修的完整原型流程。

当前比较关键的一个设计是 Refine Session。也就是说，用户每次 refine 后，结果不会立即覆盖最终输出，而是先更新为内存中的 preview polygon。只有用户确认以后，结果才真正保存。这样可以把交互逻辑和模型推理解耦，也为后续的版本管理、撤销和异常回退提供基础。

目前系统的主要问题已经不只是模型能力本身，而更多集中在交互优化策略上。比如 Auto 结果边界不规则，这是可以接受的，因为它本来就是候选检测模块。更值得关注的是 Prompt refine 尤其是 negative point refine 的稳定性。过去我们遇到过外部负点导致轮廓变化、内部负点被误判为外部、多轮 negative refine 后 mask 被切碎、polygon 变成杂乱线集合等问题。针对这些问题，当前系统已经开始加入一些工程约束，例如负点有效性判断、主 connected component 保留、mask 转 polygon 前只保留主轮廓，以及 invalid refine rejection，避免异常结果覆盖上一版有效轮廓。

下一步，我们准备重点继续完善 negative point refine，增强 polygon 质量评价机制，增加最佳结果保留能力，并逐步实现多个 tile 结果融合，最终输出可直接用于 GIS 的完整建筑矢量结果。

总体来说，当前项目已经完成了交互式建筑轮廓提取系统的主流程搭建，后续工作的重点将从“能不能提取”转向“如何稳定、可靠、可解释地完成多轮交互优化”。
