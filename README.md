**LocallyWeightedFusion** is an ensemble method based on the [scikit-learn](http://scikit-learn.org/) BaseEstimator class. It makes weighted predictions based on the qualities of the models in the ensemble on training data nearest to the test data being evaluated.


Usage
===

In Python:

```python
from LocallyWeightedFusion import LocallyWeightedFusion
# initialize classifier
lwf = LocallyWeightedFusion()
# train classifier
lwf.fit(X,Y)
# make predictions on test set
y_test = lwf.predict(X_test)
```

From the terminal:

```bash
python -m LocallyWeightedFusion.LocallyWeightedFusion data.csv
```

Acknowledgments
===
It is an implemenation of the algorithm described in *Xue, Feng, Raj Subbu, and Piero Bonissone. “Locally Weighted Fusion of Multiple Predictive Models.” In The 2006 IEEE International Joint Conference on Neural Network Proceedings, 2137–2143. IEEE, 2006. http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1716375.*

This method is being developed to study the genetic causes of human disease in the [Epistasis Lab at UPenn](http://epistasis.org). Work is partially supported by the [Warren Center for Network and Data Science](http://warrencenter.upenn.edu).  
