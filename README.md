# TSVM
这里要首先感谢我的师姐和另一个上传了TSVM的方法的人(这里贴出原链接)，他们基于书上的逻辑用python实现了TSVM，但是我最近使用时发现手动调参可能是比较令人头疼的一件事，所以我想到能不能改进一下，创建了一个TSVM方法的estimator(基于sklearn)，这样就可以通过GridSearchCV来搜索最佳的参数，于是便有了初版的代码。如果有问题交流的话可联系sunzhihan20@mails.ucas.ac.cn

#### Note：
为什么叫初版是因为这个目前还不够完善，仍然需要输入之前把Label先调成{-1, +1}。最近暂无时间，后面会根据unique_labels来写一个代替这一步操作，请期待后续更新！

## SVM介绍

**半监督支持向量机**(Semi-Supervised Support Vector Machine，简称**S3VM**) 是支持向量机在半监督学习上的推广。在**不考虑未标记样本**时，支持向量机试图找到最大隔间划分超平面，而**在考虑未标记样本**后，S3VM试图找到能将两类有标记样本分开，且穿过数据低密度区域的划分超平面，如图所示，这里的基本假设是"低密度分隔" (low-density separation)，显然，这是聚类假设在考虑了线性超平面划分后的推广。

![image](https://user-images.githubusercontent.com/88269254/174058659-26328256-7f13-44b7-9c1b-9da9c4b25d0c.png)

半监督支持向量机中最著名的是**TSVM**(Transductive Support Vector Machine)[Joachims, 1999]。与标准SVM 一样，TSVM也是针对二分类间题的学习方法. TSVM 试图考虑对未标记样本进行各种可能的标记指派(label assignment) ，即尝试将每个未标记样本分别作为正例或反例然后在所有这些结果中，寻求一个在所有样本(包括**有标记样本**和**进行了标记指派的未标记样本**)上间隔最大化的划分超平面。一旦划分超平面得以确定，未标记样本的最终标记指派就是其预测结果。

![image](https://user-images.githubusercontent.com/88269254/174058739-54d6fea2-fce2-4f5c-9aeb-250e0348cb34.png)

以上摘自西瓜书，但并不是全部，感兴趣的可以去书中查找Chapter13半监督SVM。

