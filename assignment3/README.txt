1. 系统环境

   系统：Ubuntu14.04.5 64位服务器， 内存256GB，Intel(R) Xeon(R) CPU E5-2660 v3 @ 2.60GHz，40个逻辑CPU
   IDE：PyCharm 2016.2
   编程语言：Python 3.4.3 
   使用到的第三方Python库 ： numpy:1.11.2，pandas:0.19.0, scipy:0.18.1
   数据：原始数据在项目根目录下的data目录下，文件夹为Clustering，里面含两个文件：german.txt，mnist.txt

2. 运行

   1): 进入到code根目录,可看到项目文件
   2): 在终端中运行：python3 runAssignment3.py,等待一段时间左右后(多核并发，大数据比较耗时，上述服务器运行约一个小时)，待运行结束后便可以在终端上看到结果。
   3): 结果类似如下：
        file: german, 算法: K-Medoids, 第10次运行结果:
        result: purity:0.700000, gini:0.419582, center_points:[478 127]
        file: german, 算法: K-Medoids, 第8次运行结果:
        result: purity:0.700000, gini:0.419871, center_points:[ 97 651]
        file: german, 算法: K-Medoids, 第6次运行结果:
        result: purity:0.700000, gini:0.419498, center_points:[651 694]
        file: german, 算法: K-Medoids, 第2次运行结果:
        result: purity:0.700000, gini:0.412235, center_points:[101 127]
        file: german, 算法: K-Medoids, 第4次运行结果:
        result: purity:0.700000, gini:0.419871, center_points:[ 97 651]
        file: german, 算法: K-Medoids, 第5次运行结果:
        result: purity:0.700000, gini:0.416804, center_points:[512 248]
        file: german, 算法: K-Medoids, 第3次运行结果:
        result: purity:0.700000, gini:0.411326, center_points:[155 127]
        file: german, 算法: K-Medoids, 第9次运行结果:
        result: purity:0.700000, gini:0.419907, center_points:[279 127]
        file: german, 算法: K-Medoids, 第7次运行结果:
        result: purity:0.700000, gini:0.416206, center_points:[227 127]
        file: german, 算法: K-Medoids, 第1次运行结果:
        result: purity:0.700000, gini:0.407687, center_points:[365 127]
        file: german, 算法: Spectral-Clustering-knn-3, 第1次运行结果:
        result: purity:0.700000, gini:0.419239, center_points:[696 819]
        file: german, 算法: Spectral-Clustering-knn-3, 第4次运行结果:
        result: purity:0.700000, gini:0.419239, center_points:[589 696]
        file: german, 算法: Spectral-Clustering-knn-3, 第3次运行结果:
        result: purity:0.700000, gini:0.419239, center_points:[228 173]
        file: german, 算法: Spectral-Clustering-knn-3, 第2次运行结果:
        result: purity:0.700000, gini:0.419239, center_points:[819 696]
        file: german, 算法: Spectral-Clustering-knn-6, 第3次运行结果:
        result: purity:0.700000, gini:0.418204, center_points:[223 420]
        file: german, 算法: Spectral-Clustering-knn-6, 第1次运行结果:
        result: purity:0.700000, gini:0.417718, center_points:[479 420]
        file: german, 算法: Spectral-Clustering-knn-6, 第2次运行结果:
        result: purity:0.700000, gini:0.417718, center_points:[479 420]
        file: german, 算法: Spectral-Clustering-knn-6, 第4次运行结果:
        result: purity:0.700000, gini:0.418204, center_points:[223 420]
        file: german, 算法: Spectral-Clustering-knn-9, 第1次运行结果:
        result: purity:0.700000, gini:0.418907, center_points:[588 216]
        file: german, 算法: Spectral-Clustering-knn-9, 第4次运行结果:
        result: purity:0.700000, gini:0.418907, center_points:[588 216]
        file: german, 算法: Spectral-Clustering-knn-9, 第2次运行结果:
        result: purity:0.700000, gini:0.418966, center_points:[ 48 216]
        file: german, 算法: Spectral-Clustering-knn-9, 第3次运行结果:
        result: purity:0.700000, gini:0.418907, center_points:[216 588]
        file: mnist, 算法: K-Medoids, 第7次运行结果:
        result: purity:0.394800, gini:0.741849, center_points:[9621 4978 8970 1220 2603  300 9449 4294 9943 1472]
        file: mnist, 算法: K-Medoids, 第3次运行结果:
        result: purity:0.476500, gini:0.667562, center_points:[3216 4294 1835 6520 7887 5577 6796 1430  700 9704]
        file: mnist, 算法: K-Medoids, 第8次运行结果:
        result: purity:0.448700, gini:0.712614, center_points:[9413 8739 1429 3526 5387 7103 4541  148 9291 8591]
        file: mnist, 算法: K-Medoids, 第4次运行结果:
        result: purity:0.414100, gini:0.722747, center_points:[ 450 9510 9526  873 9232 2636  496 1355 4091 3016]
        file: mnist, 算法: K-Medoids, 第10次运行结果:
        result: purity:0.514600, gini:0.646426, center_points:[5761 3834  148 8714 4920 7016 7656 6922 5972 1336]
        file: mnist, 算法: K-Medoids, 第6次运行结果:
        result: purity:0.496400, gini:0.657375, center_points:[ 106 6864 5941 8163 3407 7581 4651 9466 1472 1325]
        file: mnist, 算法: K-Medoids, 第5次运行结果:
        result: purity:0.369100, gini:0.765944, center_points:[6003 9771 4848 4278 4091 3834 2193 8658 5955 1355]
        file: mnist, 算法: K-Medoids, 第9次运行结果:
        result: purity:0.511100, gini:0.639545, center_points:[9275 7690 4019 3688 6864 9722 1336 4150 2723  422]
        file: mnist, 算法: K-Medoids, 第2次运行结果:
        result: purity:0.481600, gini:0.659028, center_points:[3704 6928 9786 3111 3681 1531  529 9870 5356 6203]
        file: mnist, 算法: K-Medoids, 第1次运行结果:
        result: purity:0.498700, gini:0.658303, center_points:[7103  700 4670 3704 5917 6681 5326 1531 9322 5556]
        ...
