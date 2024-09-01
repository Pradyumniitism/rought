### 1. **Linear Regression**
   - **Use:** Predict continuous values.
   - **When to Use:** When you need to model the relationship between a dependent variable and one or more independent variables.
   - **Code:** 
     ```python
     from sklearn.linear_model import LinearRegression
     model = LinearRegression()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 2. **Logistic Regression**
   - **Use:** Binary classification.
   - **When to Use:** When the target variable is binary (e.g., yes/no, true/false).
   - **Code:** 
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 3. **Decision Trees**
   - **Use:** Classification & regression.
   - **When to Use:** When you want an interpretable model that works well with categorical and continuous data.
   - **Code:** 
     ```python
     from sklearn.tree import DecisionTreeClassifier
     model = DecisionTreeClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 4. **Random Forest**
   - **Use:** Classification & regression.
   - **When to Use:** When you need a robust model that reduces overfitting by averaging multiple decision trees.
   - **Code:** 
     ```python
     from sklearn.ensemble import RandomForestClassifier
     model = RandomForestClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 5. **Support Vector Machine (SVM)**
   - **Use:** Classification.
   - **When to Use:** When your data has clear margin separation or when working with high-dimensional data.
   - **Code:** 
     ```python
     from sklearn.svm import SVC
     model = SVC()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 6. **K-Nearest Neighbors (KNN)**
   - **Use:** Classification.
   - **When to Use:** When you want a simple, instance-based learning algorithm and your dataset is small.
   - **Code:** 
     ```python
     from sklearn.neighbors import KNeighborsClassifier
     model = KNeighborsClassifier(n_neighbors=3)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 7. **Naive Bayes**
   - **Use:** Classification.
   - **When to Use:** When you need a fast and simple probabilistic classifier, especially with text data.
   - **Code:** 
     ```python
     from sklearn.naive_bayes import GaussianNB
     model = GaussianNB()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 8. **K-Means Clustering**
   - **Use:** Clustering.
   - **When to Use:** When you need to group data points into a predefined number of clusters based on their features.
   - **Code:** 
     ```python
     from sklearn.cluster import KMeans
     model = KMeans(n_clusters=3)
     model.fit(X)
     labels = model.predict(X)
     ```

### 9. **Gradient Boosting**
   - **Use:** Classification & regression.
   - **When to Use:** When you need a strong predictive model that iteratively improves by correcting errors of the previous models.
   - **Code:** 
     ```python
     from sklearn.ensemble import GradientBoostingClassifier
     model = GradientBoostingClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 10. **AdaBoost**
   - **Use:** Classification & regression.
   - **When to Use:** When you want to combine multiple weak learners to form a strong learner.
   - **Code:** 
     ```python
     from sklearn.ensemble import AdaBoostClassifier
     model = AdaBoostClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 11. **XGBoost**
   - **Use:** Classification & regression.
   - **When to Use:** When you need an efficient, scalable, and flexible gradient boosting model.
   - **Code:** 
     ```python
     import xgboost as xgb
     model = xgb.XGBClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 12. **Lasso Regression**
   - **Use:** Regression with L1 regularization.
   - **When to Use:** When you want to reduce model complexity and enforce sparsity in the coefficients.
   - **Code:** 
     ```python
     from sklearn.linear_model import Lasso
     model = Lasso(alpha=0.1)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 13. **Ridge Regression**
   - **Use:** Regression with L2 regularization.
   - **When to Use:** When you need to prevent overfitting in a linear regression model.
   - **Code:** 
     ```python
     from sklearn.linear_model import Ridge
     model = Ridge(alpha=1.0)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 14. **ElasticNet**
   - **Use:** Regression combining L1 and L2 regularization.
   - **When to Use:** When you need a balance between Lasso and Ridge regression.
   - **Code:** 
     ```python
     from sklearn.linear_model import ElasticNet
     model = ElasticNet(alpha=0.1)
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 15. **LightGBM**
   - **Use:** Classification & regression.
   - **When to Use:** When you need a highly efficient and scalable gradient boosting model for large datasets.
   - **Code:** 
     ```python
     import lightgbm as lgb
     model = lgb.LGBMClassifier()
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 16. **DBSCAN**
   - **Use:** Clustering.
   - **When to Use:** When you want to find clusters in data with varying densities and identify outliers.
   - **Code:** 
     ```python
     from sklearn.cluster import DBSCAN
     model = DBSCAN(eps=3, min_samples=2)
     labels = model.fit_predict(X)
     ```

### 17. **PCA (Principal Component Analysis)**
   - **Use:** Dimensionality reduction.
   - **When to Use:** When you want to reduce the dimensionality of your data while retaining the most variance.
   - **Code:** 
     ```python
     from sklearn.decomposition import PCA
     pca = PCA(n_components=1)
     X_pca = pca.fit_transform(X)
     ```

### 18. **t-SNE**
   - **Use:** Visualization of high-dimensional data.
   - **When to Use:** When you need to visualize complex, high-dimensional datasets in 2D or 3D.
   - **Code:** 
     ```python
     from sklearn.manifold import TSNE
     model = TSNE(n_components=2)
     X_tsne = model.fit_transform(X)
     ```

### 19. **Gaussian Mixture Model (GMM)**
   - **Use:** Clustering.
   - **When to Use:** When you want to model data as a mixture of several Gaussian distributions.
   - **Code:** 
     ```python
     from sklearn.mixture import GaussianMixture
     model = GaussianMixture(n_components=2)
     model.fit(X)
     labels = model.predict(X)
     ```

### 20. **Support Vector Regression (SVR)**
   - **Use:** Regression.
   - **When to Use:** When you need to predict continuous outcomes and your data has complex, non-linear relationships.
   - **Code:** 
     ```python
     from sklearn.svm import SVR
     model = SVR(kernel='linear')
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
     ```

### 21. **Autoencoders**
   - **Use:** Dimensionality reduction, anomaly detection.
   - **When to Use:** When you need to learn efficient data representations or perform tasks like anomaly detection.
   - **Code:** 
     ```python
     import tensorflow as tf
     input_layer = tf.keras.layers.Input(shape=(2,))
     encoded = tf.keras.layers.Dense(1, activation='relu')(input_layer)
     decoded = tf.keras.layers.Dense(2, activation='sigmoid')(encoded)
     autoencoder = tf.keras.models.Model(input_layer, decoded)
     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
     ```

### 22. **Reinforcement Learning (RL)**
   - **Use:** Maximizing cumulative reward in an environment.
   - **When to Use:** When you're working with an environment where an agent must learn to make decisions to achieve a goal.
   - **Code:** 
     ```python
     def update_q_table(state, action, reward, next_state):
     ```
