SAMPoly AGENTS.md v2.0
======================

* * *

Project Overview
================

当前项目基于 SAMPolyBuild。

项目目标：

实现面向遥感建筑物的交互式轮廓提取系统。

长期规划包括：

1. 单点自适应 BBox 生成

2. 正负点交互优化

3. Polygon 实时修正

4. 建筑轮廓规则化

5. GIS 导出

6. QGIS 插件集成

* * *

Rule 0: Understand Before Coding
================================

任何开发任务开始后：

禁止直接修改代码。

必须先完成：

1. 理解需求

2. 分析当前实现

3. 梳理调用链

4. 输出修改方案

获得用户确认后才能开始编码。

* * *

Rule 1: CodeGraph First
=======================

如果项目已接入 CodeGraph：

优先使用：

* codegraph_explore

* codegraph_search

* codegraph_callers

* codegraph_callees

* codegraph_impact

分析：

* 推理入口

* 数据流

* 模块依赖

* 修改影响范围

禁止直接全项目 grep。

* * *

Rule 2: Project Architecture Protection
=======================================

项目分为：
Layer 1
-------

模型层

包括：

* SAMPoly

* SAM

* Encoder

* Decoder

* * *

Layer 2
-------

推理层

包括：

* Predictor

* Inference

* PostProcess

* * *

Layer 3
-------

交互层

包括：

* BBox输入

* Point输入

* Polygon优化

* GUI交互

* * *

开发优先修改：
    Layer 3

其次：
    Layer 2

除非用户明确要求：

禁止修改：
    Layer 1

* * *

Rule 3: Model Protection
========================

禁止：

* 修改模型结构

* 修改网络层定义

* 修改训练策略

* 修改损失函数

* 修改预训练权重

除非用户明确要求。

* * *

Rule 4: Backward Compatibility
==============================

必须保留：

* 原始BBox模式

* 原始推理流程

* 原始测试流程

新增功能必须向后兼容。

* * *

Rule 5: Git Policy
==================

任务开始前：
    git status

* * *

禁止直接在：
    main
    master

开发。

* * *

新任务必须从：
    main/master

创建：
    feature/<task-name>

* * *

禁止：

从已有 feature 分支继续创建新 feature 分支。

* * *

未经用户确认：

禁止：
    git merge
    git rebase
    git push
    git reset --hard
    git clean -fd

* * *

每个阶段结束：

必须提交 commit。

Commit 前缀：
    feat
    fix
    refactor
    docs
    test

* * *

Rule 6: Minimal Modification Principle
======================================

优先：

* 复用已有代码

* 扩展已有逻辑

避免：

* 大范围重构

* 推翻现有流程

* 重复实现

坚持最小修改原则。

* * *

Rule 7: Configuration Management
================================

禁止：

* 写死路径

* 写死权重文件

* 写死服务器地址

所有配置：

统一管理。

* * *

Rule 8: Error Handling
======================

禁止：

静默失败。

必须：

* 抛出明确异常

* 输出明确日志

* 提供错误原因

* * *

Rule 9: Interaction Logic
=========================

新增交互功能时：

必须与模型推理解耦。

要求：
    Interaction
    ↓
    Predictor
    ↓
    Model

禁止：
    Interaction
    ↓
    直接修改Model

* * *

Rule 10: Testability
====================

新增功能：

必须支持：

* 无GUI测试

* 命令行测试

* 自动化测试

测试逻辑不得依赖界面。

* * *

Rule 11: Research Reproducibility
=================================

任何实验性功能：

必须说明：

1. 修改原因

2. 理论依据

3. 验证方法

4. 预期效果

禁止：

凭经验修改算法。

* * *

Rule 12: Polygon Validation
===========================

涉及 Polygon 修改：

必须验证：

* 顶点数量

* 闭合性

* 坐标合法性

* 输出格式

防止产生非法轮廓。

* * *

Rule 13: Linux Runtime Environment
==================================

项目源码：
    Windows

实际运行环境：
    Linux Remote Server

Conda环境：
    sampoly

* * *

默认情况下：

不要执行 Python 测试。

不要假设：

* CUDA存在

* PyTorch可运行

* 本机具有运行环境

* * *

应生成：

* Linux测试命令

* 预期输出

* 调试建议

等待用户执行。

* * *

Rule 14: Documentation
======================

新增功能后：

同步更新：

* README

* 使用说明

* 参数说明

* 测试说明

* * *

Rule 15: Task Report
====================

任务结束后：

必须输出：
Modified Files
--------------

修改文件
Summary
-------

修改内容
Linux Test Commands
-------------------

测试命令
Expected Results
----------------

预期结果
Risks
-----

风险分析
Next Suggestions
----------------

下一步建议

* * *

Rule 16: Research Project Priority
==================================

本项目属于：

* 科研项目

* 毕业论文项目

* AI建筑工具原型项目

优先保证：

1. 正确性

2. 可复现性

3. 可解释性

高于：

* 性能优化

* 代码风格优化

* 架构重构

* * *

Rule 17: Completion Definition
==============================

只有满足：

* 修改完成

* 调用链分析完成

* 测试命令提供完成

* 验证方案给出

* 风险已说明

* 文档已更新（如需要）

才能宣布：

Task Completed
