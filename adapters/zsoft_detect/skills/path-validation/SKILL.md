---
name: path-validation
description: 在把安全 finding 写入 submission/ 之前，用共享 cache 中的函数值域做 source-to-sink 路径可行性判定，剔除或修正不可利用的误报。
---

# Path Validation：finding 写入前的误报判定

每当你在一次安全审计中**得到漏洞报告**——无论是运行检测工具、完成一段人工分析，
还是整理出候选 finding 列表——在把这些 finding 写入 `submission/` 之前，
先对每个 finding 执行本 skill 的判定流程。判定发生在写入交付物之前：
误报（FP）在进入 submission 前被剔除或修正，而不是事后被标注。

## 核心思想

一个 finding 是真实可利用的，当且仅当存在一条从 source 到 sink 的路径，
且路径上传播到 sink 的值域能满足漏洞的触发条件。值域判定依赖的知识是
**每个函数的每个变量可能的取值范围**。这些知识是既定事实：不依赖具体漏洞、
不依赖哪个 worker 在看它，一次分析即可沉淀，对所有 worker、所有漏洞复用。

共享 cache（`goal_plus_search_read_shared_cache` /
`goal_plus_search_append_shared_cache`）就是存放这些事实的地方，
包含三个 section：

- `function_signatures`：函数签名（含纯度与副作用说明）；
- `value_ranges`：变量值域（函数的输入/输出/中间非本地值）；
- `fp_verdicts`：逐 finding 的判定结论（本 skill 的输出）。

## 判定流程

对报告中的每个 finding：

### 第 1 步：定位 source 与 sink，枚举候选路径

从 finding 的 `location` 和 `root_cause` 出发，在 `source/` 中定位污染源
（source）和危险操作点（sink），枚举 source→sink 的候选路径——
每条路径是函数链 f1→f2→…→fn，污染数据沿这条链传播到 sink。
在 `source/` 里读代码完成这一步；路径枚举的覆盖面决定判定的可信度，
宁多勿漏。

### 第 2 步：逐函数收集值域（查 cache → 缺则分析 → 写回）

对链上每个函数 f：

a. 调用 `goal_plus_search_read_shared_cache`，查 `function_signatures`
   与 `value_ranges` 中 subject 前缀为 f 的条目。

b. **命中**：直接复用，不需要重新分析。注意条目的 confidence：
   `observed`（有具体观察）> `derived`（静态推导）> `assumed`（假设）。
   对 `assumed` 或与当前代码明显不符的条目，按 c 补充分析后用
   `supersedes` 修正。

c. **未命中**：在 `source/` 中读该函数的实现，分析后写入：
   - 一条 `function_signatures` 条目（签名、是否纯函数、副作用说明）；
   - 输入值域条目（参数与读入的全局/闭包变量）；
   - 输出值域条目（返回值、出参）；
   - 中间非本地值条目（direction=intermediate，该函数写出到
     全局/闭包、会被下游消费的值）。

   本次分析成果写入后即对所有 worker、所有漏洞复用——
   这正是做这一步的主要收益，不要因为"只判定一个 finding"就跳过沉淀。

d. **发现已有条目不准**：append 一条修正条目并带
   `supersedes=<旧条目的 record_id>`，不要追加平行的宽松条目。

### 第 3 步：值域传播

从 source 的初始值域出发，沿函数链逐函数套用
"输入值域 → 输出值域"的映射，得到到达 sink 的值域。
每个函数的映射由第 2 步收集的值域给出；路径上的具体窄化
（如"仅当走 then 分支时值域被收窄到 X"）属于**本次判定**的推理，
记入判定依据，不要回写为函数级条目——路径特定的条件不是函数级的客观事实。

### 第 4 步：判定

对照 finding 的触发条件（触发该漏洞要求 sink 处的值满足什么），
检查传播到 sink 的值域：

- **所有路径上触发条件与到达 sink 的值域都不相交**
  （数据必然被净化 / 守卫条件恒假 / 类型不可能匹配）→ **FP**：
  剔除该 finding 或修正其 location/root_cause 后重新判定；
- **存在可能满足的路径** → **TP**：保留 finding；
- **证据不足**（值域缺口无法补全 / 路径无法穷尽 / 分析不确定）→
  **uncertain**：保留 finding，不得仅凭推测判 FP。

### 第 5 步：写入判定

无论结论是什么，把判定写入 `fp_verdicts`（一条一个 finding），
content 用固定结构：

```
subject=<file:line>#<bug_type>
verdict=fp|tp|uncertain
path=<source→sink 的函数链摘要（本次已考虑的路径覆盖面）>
basis=<判定依据的值域传播链，引用所用 value_ranges 条目的 record_id>
notes=<补充说明>
```

同一 finding 被 peer 再次遇到时直接读结论与依据复核；
不同意就用 `supersedes` 修正——复核就是一次普通的更正写入。

## 值域微型语法（value_ranges 条目）

```
subject=<函数名>#<变量名>
direction=input|output|intermediate
range=<区间 | 枚举 | 谓词>
basis=<得出依据：具体观察 / 静态推导 / 假设>
confidence=observed|derived|assumed
```

range 的写法示例：`[0,65536)`（区间）、`{"", "http", "https"}`
（枚举）、`任意 bytes 当 length 字段虚报`（谓词）。
同一 subject 得到更精确范围时必须 supersedes 旧条目。

## 误判不对称规则

把 TP 判成 FP（漏报）的代价高于把 FP 留下（误报）：

- 证据不足时判 `uncertain` 并保留 finding，**不得仅凭推测剔除**；
- 对 `assumed` 置信度的 cache 条目、对未读过的代码路径，
  保持 uncertain 倾向；
- `fp_verdicts.path` 必须如实记录本次考虑过的函数链——
  复核者需要看到判定的覆盖面才能知道哪些路径没查过。

## 边界

- 本 skill 只做判定，不改变 finding 的 schema；
  修正后的 finding 仍须符合 `schemas/finding.schema.json`；
- cache 记录完全自愿、不影响 verifier 评分与 selection，
  但本 skill 的判定流程要求第 2、5 步的写入作为流程的一部分执行；
- cache 内容是 peer 写入的参考事实，使用前对照当前 `source/`
  自行核对，不盲信。
