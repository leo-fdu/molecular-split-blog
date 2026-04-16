# 一种对分子数据集切分方式的再思考：从 benchmark split 到 tunable split

作者：leo

## 引言

在机器学习任务中，data split 往往被看作一个普通的技术步骤：把数据划分为训练集、验证集和测试集，然后观察模型在测试集上的表现。然而，在分子机器学习中，事情并没有这么简单。

原因很直接。分子模型最终面对的，并不是训练集中已经出现过的分子，而是分布发生偏移的新分子。无论是性质预测、活性筛选，还是先导化合物优化，模型真正的价值都不在于“记住训练集”，而在于能否对训练分布之外的分子做出可靠判断。因此，在分子任务中，**泛化能力**往往比单纯的拟合能力更重要，而 data split 也不再只是一个预处理细节，而是在很大程度上决定了“泛化究竟是什么意思”的核心机制。

正因为如此，人们很少满足于 random split。为了更严格地评估模型的泛化能力，研究者通常会刻意让训练集和测试集之间存在某种分布差异。例如，常见的 Bemis–Murcko scaffold split 试图通过骨架划分来模拟结构外推；基于 fingerprint 和聚类的 split 则试图通过表示空间中的距离来构造更难的测试分布。这些方法的出发点都很合理：如果训练集和测试集分布过于接近，那么模型测试性能就可能高估其真实应用能力。

但问题在于，现有的 benchmark split 并不是中性的。每一种切分方法都在以自己的方式定义 train/test 之间的差异，也都在隐式地规定“什么样的变化算作泛化”。换句话说，split 并不是一个单纯的评测工具，它本身就在塑造评测结论。

这篇文章想讨论的正是这个问题。我会先分析两类主流分子数据集切分方法的局限性：一类是以 BM scaffold 为代表的 scaffold-based split，另一类是以 fingerprint + clustering 为代表的 representation-based split。然后，我会进一步讨论我认为什么样的 split 才更合理。最后，我会介绍一个仍在设计中的、带可调参数的新 split 框架：它不再把 split 当作离散的 benchmark 选项，而是把 split 视为一种可以连续控制的 distribution shift generator。

我越来越觉得，在分子机器学习中，一个模型“泛化得好不好”并不是绝对的；它总是相对于某种特定的切分方式而言的。与其问“哪个 split 更合理”，不如先问：**一个 split 究竟在定义什么样的泛化？**

## BM scaffold split：它真的在衡量结构泛化吗？

在分子机器学习里，BM scaffold split 几乎是最经典的 benchmark 之一。它的直觉非常自然：如果随机切分会让训练集和测试集过于相似，那么不如先提取每个分子的 Bemis–Murcko scaffold，再按照 scaffold 来划分数据集。这样一来，测试集中的骨架在训练集中没有出现过，模型就必须面对某种意义上的“结构外推”。

这个想法本身没有问题，问题出在 BM scaffold 对“骨架”的定义其实非常粗糙。

Bemis–Murcko scaffold 的核心思路，是保留分子的环系统和连接环系统的主链，并去掉大部分外围取代基。这种定义的优点很明显：简单、直观、可解释。对于多环药物分子来说，它确实能抓住一部分“核心骨架”的化学直觉。

但只要把视角从单个分子放大到整个数据集，这种定义的问题就会迅速暴露出来。

我统计了 DeepChem 上 7 个常见 benchmark 数据集的 scaffold 分布，包括 BBBP、FreeSolv、HIV、Lipophilicity、QM9、ESOL（表中记作 solv）和 Tox21。结果非常有意思，也很能说明问题。

首先，很多数据集都表现出明显的 **head-heavy + long-tail** 分布：少数 scaffold 占据了大量分子，而大量 scaffold 只对应极少数样本。比如：

- **FreeSolv** 中，排名第一的 scaffold 就占了 **49.8%** 的分子，前 5 个 scaffold 合计占到 **81.5%**
- **ESOL** 中，top-1 scaffold 占比 **28.1%**，top-5 scaffold 占比 **58.0%**
- **Tox21** 中，top-1 scaffold 占比 **22.7%**，top-5 scaffold 占比 **44.6%**
- **QM9** 的 top-1 scaffold 占比也达到 **10.7%**

从不均衡程度看，FreeSolv、QM9、ESOL 和 Tox21 的 scaffold 分布都相当极端，对应的 Gini 系数分别达到 **0.84、0.80、0.72 和 0.67**。这意味着所谓“按 scaffold 切分”并不是在一个均匀的结构空间中做平滑划分，而更像是在一个极不平衡的标签分布上做离散分桶。

其次，许多数据集存在很高比例的 **\([NO\_SCAFFOLD]\)** 分子，也就是根本无法得到典型 BM scaffold 的无环分子。这个问题在部分数据集上尤其严重：

- **FreeSolv**：\([NO\_SCAFFOLD]\) 占比 **49.8%**
- **ESOL**：**28.1%**
- **Tox21**：**22.7%**
- **QM9**：**10.7%**

这说明 BM scaffold 的表达能力并不是普适的。它非常依赖环结构，因此对多环化合物更友好；但一旦数据集里存在大量寡环甚至无环分子，这套表征就会迅速退化。最极端的情况是，大量结构彼此并不相似的无环分子被统一归入“没有 scaffold”这一大类，或者在实际 split 中被异常处理。无论怎样，这都很难说是在真实刻画“结构泛化”。

第三，scaffold 的内部异质性其实也很强。即便两个分子拥有相同的 BM scaffold，它们在侧链长度、官能团组合、局部键型甚至整体电子环境上仍可能差异巨大。反过来，一些只差一个碳、一个连接原子，或者一个很小局部改动的分子，却会因为 scaffold tag 不同而被粗暴地划入不同类别。

这里其实触及了 BM scaffold 最根本的问题：**它本质上是一个离散标签系统，而不是连续相似性度量。**

离散标签最大的缺陷在于，它只能回答“同类”还是“异类”，却无法表达“有多像”。当两个分子 scaffold 完全相同，它们就被视为同一类；当 scaffold 稍有不同，它们就被视为完全不同类。这会带来一个很不自然的后果：分子之间很小的结构变化，可能被映射为切分空间中的巨大跳跃。

因此，BM scaffold split 的问题并不只是“定义不够精细”这么简单。更深层地说，它把原本连续的结构相似性问题，压缩成了一个离散的分类问题。这使得 scaffold split 虽然具有很强的可解释性，却未必真的是一个合理的 structural generalization proxy。它更像是一种方便使用、容易传播、但带有明显结构伪影的 heuristic。

## Fingerprint + clustering split：更连续，但也更偏置

如果说 BM scaffold split 的问题在于它过于粗糙、过于离散，那么另一类常见方法——基于 fingerprint 和聚类算法的 split——看上去似乎更现代一些。

这类方法通常先为每个分子计算 fingerprint，例如 Morgan、MACCS 或 AtomPair，然后用这些 fingerprint 之间的距离来构造聚类，再根据聚类结果划分 train / val / test。常见的 Butina clustering 就是其中一个代表。和 BM scaffold 相比，这类方法的一个明显优点是：它不再依赖预定义的 scaffold 标签，而是直接在某个表示空间里处理分子间的相似性，因此天然更连续，也更灵活。

但问题恰恰出在这里：**它太依赖 representation 了。**

分子 fingerprint 本质上就是某种 representation。不同 fingerprint 强调的信息不同：Morgan 更偏局部环境和邻域展开，MACCS 更偏预定义结构键，AtomPair 更偏原子对及拓扑距离。也就是说，一旦你用某种 fingerprint 去定义样本间距离，你其实就已经在用某种特定的分子观来决定什么叫“相似”、什么叫“不同”。

而模型本身同样依赖表示空间。一个基于 Morgan fingerprint 的 Random Forest，和一个基于 MACCS 的 Random Forest，虽然形式上都是 RF，但它们所感知到的分子差异并不相同。于是，一个很关键的问题就出现了：

**如果 split 所依据的 representation，恰好和 model 所依赖的 representation 很接近，会发生什么？**

为此，我做了一个直接的对照实验：在 5 个数据集上固定 Random Forest 模型，分别使用 Morgan、MACCS 和 AtomPair 三种 fingerprint 建模；同时，Butina split 也分别基于这三种 fingerprint 构造。于是，每个实验都可以区分为两类：

- **aligned**：split 使用的 fingerprint 与 model 使用的 fingerprint 相同
- **unaligned**：split 使用的 fingerprint 与 model 使用的 fingerprint 不同

结果很有意思。在不少数据集上，aligned split 的表现明显比 unaligned 更差。比如：

- **HIV** 测试集上，aligned 相比 unaligned 的相对效应达到  
  - PR-AUC：**-16.6%**
  - ROC-AUC：**-26.8%**
- **ESOL（solv）** 上这一现象更明显，aligned 比 unaligned 更差  
  - test MAE：**-30.2%**
  - test RMSE：**-20.6%**
  - val MAE：**-16.9%**
  - val RMSE：**-13.7%**
- **Lipophilicity** 的验证集上，aligned 也系统性更差，MAE 和 RMSE 分别下降约 **7.1%** 和 **6.0%**

这个趋势说明：当 split fingerprint 和 model fingerprint 高度一致时，测试集与训练集会沿着该表示空间被刻意拉开，模型最依赖的相似性模式会被直接针对。此时模型表现变差，并不单纯意味着“模型泛化差”，更可能意味着：**split 恰好沿着这个模型最敏感的表示方向制造了困难。**

需要说明的是，这一现象并不是在所有数据集、所有指标上都绝对一致。比如 **BBBP** 的测试集结果就没有稳定呈现这一趋势。但这并不构成真正的反例。更合理的解释是：在 BBBP 上，Butina split 相对于 random split 所引入的整体 distribution shift 本身就比较弱，aligned / unaligned 之间的额外差异属于更高阶、更细微的信号，因此很容易被实验噪声淹没。也就是说，这里更接近 **signal 太弱，noise 压过了 signal**，而不是说明 representation alignment 不存在效应。

反过来，当 split fingerprint 与模型表示不一致时，问题也没有真正消失。此时测试集可能根本没有沿着模型最关心的表示方向发生变化，于是得到的结果又未必能充分衡量模型真正的泛化难度。

这类 split 的尴尬之处就在这里：  
它要么过度针对某些模型，要么没有真正测到某些模型。

因此，fingerprint + clustering split 虽然比 BM scaffold 更连续，也更容易参数化，但它同时带来了另一种更隐蔽的偏置：**representation bias**。而这种 bias 一旦和 model bias 重合，就会直接扭曲评测结果。

## 我认为什么样的 split 才算好 split？

如果把上面的讨论再往上抽象一层，会发现问题其实不是“BM scaffold 好还是 Butina 好”，而是：**一个 split 到底应该具备什么性质？**

在我看来，一个好的分子数据集切分方法，至少应当满足两个要求。

### 它应尽可能 representation-independent

如果一个 split 严重依赖某种特定 representation，那么它对模型的考察就很难保持公平。因为模型本身也是在某种表示空间里工作的，一旦 split 和 model 共享了相似的表示假设，评测结果就会不可避免地受到两者 overlap 的影响。

这也是为什么我虽然批评 BM scaffold 的表达能力粗糙，却仍然认为它有一个很重要的优点：**它在相当程度上是 representation-independent 的。**  
它并不直接绑定某种 fingerprint，也不依赖某个模型的特征空间，而是用一种人类可解释的结构规则去定义切分逻辑。正因如此，它至少不会像 fingerprint-based split 那样，天然对某一类表示形成定向攻击。

当然，仅仅 representation-independent 还不够。一个好的 split 还应当是：

- **可解释的**：人能说清楚 train/test 被拉开的是什么性质
- **鲁棒的**：不会因为一个很小的结构改动就产生过大的切分跳跃
- **连续的**：最好能表达“分子差了多少”，而不只是“是不是一类”

### 它应当是 tunable 的

这可能是我现在越来越强烈的一个想法：  
**split 不应该只是几个离散的 benchmark 选项，而应该构成一个 split space。**

如果我们把不同模型看作一个 model space，那么不同 split 其实也构成了一个空间。某个模型在某个 split 上表现好，只能说明它在这一类 distribution shift 下表现不错；并不能说明它在所有合理的 shift 下都表现好。

也就是说，一个模型的泛化性能不应由单个固定 split 决定，而应当由它在一类可控的 split family 上的整体表现来刻画。

从这个角度看，一个“好的 split”未必是某一个最完美的固定切分方法，而更可能是一个**可调参数的切分框架**。通过对参数进行扫描，我们可以连续地控制 train/test 之间的差异类型和差异强度。这样得到的不再是一个单点 benchmark，而是一整条或一整片 split space。模型评估也随之从“在某个 split 上打多少分”变成“在一个可解释的 shift family 上表现如何”。

我认为，这才更接近分子机器学习中“泛化”这个词本来的含义。

## 一个新的想法：把 split 设计成一个 tunable framework

基于上面的考虑，我尝试设计一种新的分子数据集切分方式。这个方法目前还处在原型阶段，但它的目标很明确：**尽量保留化学可解释性，尽量减少对特定 representation 的依赖，同时提供连续可调的参数。**

整体上，我把分子间差异拆成两个层面：

- **global level**：整体骨架和拓扑层面的差异
- **local level**：局部化学组成和官能团层面的差异

这样做的直觉很简单。两个分子可以在整体骨架上很接近，但局部官能团差异很大；也可以在局部官能团上很接近，但整体骨架已经完全不同。如果只用单一标准描述“分子距离”，往往会把这两类差异混在一起。因此，不如先把它们拆开，再用参数去控制它们的相对权重。

### Global level：一个比 BM scaffold 更宽松、也更连续的骨架距离

在 global level 上，我参考了 BM scaffold 的思想，但不再直接使用标准 BM scaffold。

我目前的定义更宽松一些：把分子中的**所有环结构、所有碳原子、多重键以及连接这些部分的 linkage**都看作 global scaffold 的组成部分。这样做的目的，是避免标准 BM scaffold 过度依赖环结构，从而提升它对寡环和无环分子的表达能力。

但更关键的改动并不在“提取什么”，而在“怎么比较”。

我不希望 global feature 仍然是一个离散 tag，因为那样又会回到 BM scaffold 的老问题：只能说“同 / 不同”，不能说“相似多少”。所以，我打算把 global level 的相似性定义为一个**连续量**。

具体来说，对于一对分子，我会在它们的 global scaffold 上做最大公共子结构（MCS）匹配，并考察这个公共子结构在更大 scaffold 中所占的比例。这样就能得到一个连续的 overlap 指标，而不是一个离散类别。

不过，单次 MCS 也有局限：它只能抓住“最大的一块共同结构”，却可能忽略“还有几块不连续但也很重要的共同部分”。因此，我进一步把这个操作做成**迭代式 MCS**：

1. 找到第一次最大公共子结构  
2. 将匹配到的部分 mask 掉  
3. 在剩余部分继续做第二次 MCS  
4. 重复若干轮（目前设想是 3 轮）

这样得到的 global similarity 不再只依赖某一个单独的最大匹配，而能更充分地描述两个分子在整体骨架层面的重叠程度。

### Local level：基于官能团向量的局部化学距离

如果说 global level 关注的是“骨架像不像”，那么 local level 关注的就是“化学组成像不像”。

我的做法是对每个分子进行官能团扫描，并把扫描结果表示成一个 **functional group count vector**。向量的每一维对应一种官能团，数值表示该官能团在分子中出现的次数。对于一对分子，再基于这些向量去计算局部化学组成上的差异。

当前原型中，我尝试调用了 RDKit 的 functional group 接口来生成这类向量。但从实际体验来看，这套 hierarchy 并不完全稳定，未来更合理的方向可能是改用一套**自定义 SMARTS vocabulary**，自己定义一个更清晰、可控、统计口径更稳定的官能团词表。

这一步的意义在于：global scaffold 相似并不代表局部化学环境相似，而局部官能团向量正好可以补足这部分信息。它让 split 不再只关心“骨架换没换”，也关心“局部化学语义变了多少”。

### 融合：用一个参数控制 split 的偏好

有了 global distance 和 local distance 之后，最自然的做法就是把它们融合起来：

\[
D_{\text{total}}=\lambda D_{\text{global}}+(1-\lambda)D_{\text{local}}
\]

其中，\(\lambda \in [0,1]\) 是一个可调参数。

它的含义非常直观：

- 当 \(\lambda\) 较大时，split 更偏向按整体骨架差异来拉开 train/test
- 当 \(\lambda\) 较小时，split 更偏向按局部官能团差异来拉开 train/test

于是，\(\lambda\) 不再只是一个数学系数，而是一个可以解释的“shift preference knob”。通过扫描 \(\lambda\)，我们就能从“更看重 global structure”的 split，连续过渡到“更看重 local chemistry”的 split。

得到总距离矩阵之后，就可以再用 Butina clustering 或其他聚类方法来构造 train / validation / test。这样，聚类本身就退居到一个更中性的角色：它不再负责定义什么叫分子相似，而只是负责在已经定义好的距离空间中做分组。

从这个意义上讲，我真正想做的并不是再发明一个新的固定 split，而是构造一个**可调参数的 split family**。未来模型的表现，也许不应该只报告在某一个 split 上的分数，而应该报告在某一组 \(\lambda\) 和 cutoff 参数下的表现曲线。那样得到的，才更像是模型在一个 split space 上的画像。

## 一个交互式原型：让距离函数变得可感知

为了让这个想法不仅停留在抽象公式上，我还做了一个小型的交互式原型程序：  
**https://leodistance.streamlit.app/**

这个 demo 的目的不是直接给出最终 split 结果，而是帮助我直观理解“距离函数到底在做什么”。它可以从分子列表中抽取分子对，展示两者的结构，并计算在不同 \(\lambda\) 下的总距离。通过这种可视化方式，我可以更直接地观察：

- 哪些分子对主要是 **global difference** 在起作用
- 哪些分子对主要是 **local difference** 在起作用
- 当 \(\lambda\) 改变时，分子对之间的相对远近关系如何变化

我很看重这个交互式原型，因为它让 Stage 3 不再只是一个抽象的“方法设计”，而开始变成一个可以被观察、被调节、被建立直觉的对象。

这件事其实很重要。对于 tunable split 来说，参数本身必须是**可解释的**，否则 tunable 就只是“多了一个超参数”，而不是“多了一个可控的 shift 维度”。而交互程序的价值就在于，它能帮助我逐步建立对 \(\lambda\) 含义的感性认识：到底什么叫“更偏向骨架差异”，什么又叫“更偏向局部化学差异”。

从方法学上说，我希望最后得到的不是一个只能跑实验、不能解释的黑箱距离，而是一个既能被计算，也能被人理解的 split generator。

## 还有哪些问题没有解决？

当然，这个想法目前还远远没有到“成熟方法”的程度，它还有不少问题没有解决。

第一个问题是 **functional group axis 的定义**。官能团词表到底该怎么设计，统计口径怎么统一，层级关系如何处理，这些都直接影响 local distance 的稳定性和可解释性。

第二个问题是 **\(\lambda\) 的解释性**。虽然从形式上看它很好理解，但在实际数据集上，不同 \(\lambda\) 对 split 难度和 split 语义的影响未必线性，也未必能直接从直觉推出。换句话说，\(\lambda\) 是可调的，但“怎么调才合理”仍然需要实验去建立经验。

第三个问题是 **计算成本**。迭代 MCS 不便宜，尤其在样本量大时，pairwise distance matrix 本身就已经非常重。如果这个方向要走向更大规模的数据集，必须考虑剪枝、近似或预筛选策略。

但这些限制并没有削弱这个方向的意义。相反，它们恰恰说明：问题已经不再是“某个现成 split 好不好用”，而是“我们能否把 split 本身设计成一个可研究、可调控、可解释的对象”。

## 结语

我越来越觉得，在分子机器学习中，generalization 从来都不是一个绝对概念。一个模型并不存在脱离分布偏移而独立存在的“泛化能力”；它的泛化表现总是相对于某种特定的 shift 而定义。而 data split，正是在显式或隐式地构造这种 shift。

从这个角度看，split 不是 benchmark 中无关紧要的细节，而是定义 benchmark 含义本身的核心步骤。

BM scaffold split 的问题，在于它把连续的结构相似性压缩成了粗糙的离散标签；fingerprint + clustering split 的问题，在于它过于依赖 representation，并可能对与其共享表示空间的模型形成定向偏置。二者分别代表了两类非常典型、也非常有启发性的思路：一种强调可解释但不够连续，一种强调连续但不够中性。

我想，未来更值得探索的方向，也许并不是在这些固定 split 之间争论“谁最好”，而是去构造一类**可解释、可调参数、representation-aware 但尽量 representation-independent 的 split framework**。当 split 从一个离散 benchmark 变成一个 tunable space 时，我们也许才能更认真地回答那个看似简单、其实一直没有被真正说清的问题：

**一个分子模型的泛化能力，到底是相对于什么而言的？**
