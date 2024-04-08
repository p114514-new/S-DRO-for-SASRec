# S-DRO-for-SASRec
Implementing the idea of S-DRO into SASRec.
Fits into Recbole's sequential recommender module (needs to import this file into __init__.py of sequential recommender folder).
Needs to fix the program if the user id column is not named user_id or the item id column is not named item_id. (You may find from or manually include these names into config files)

S-DRO: Hongyi Wen, Xinyang Yi, Tiansheng Yao, Jiaxi Tang, Lichan Hong, and Ed H. Chi. 2022. Distributionally-robust Recommendations for Improving Worst-case User Experience. In Proceedings of the ACM Web Conference 2022 (WWW '22), April 25–29, 2022, Virtual Event, Lyon, France. ACM, New York, NY, USA 5 Pages. https://doi.org/10.1145/3485447.3512255
