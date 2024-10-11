English Summary:
This Python code analyzes an insurance dataset to predict the "charges" (insurance costs) using several regression algorithms. After loading and inspecting the data, the categorical columns like "sex," "smoker," and "region" are encoded into numerical values using one-hot encoding. The target column is "charges," and the remaining features are used for model training. The data is then split into training and testing sets and standardized.

Five regression models are trained: Decision Tree Regressor, Random Forest Regressor, Lasso, ElasticNet, and Ridge. For each model, performance metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) are calculated and displayed in a table to compare the models’ effectiveness in predicting insurance charges.

Türkçe Özet:
Bu Python kodu, sigorta verileri üzerinde "charges" (sigorta maliyeti) değerini tahmin etmek için çeşitli regresyon algoritmalarını kullanır. Veriler yüklendikten ve incelendikten sonra, "sex," "smoker" ve "region" gibi kategorik sütunlar one-hot encoding yöntemiyle sayısal değerlere dönüştürülür. Hedef sütun "charges" olup, geri kalan özellikler model eğitimi için kullanılır. Veriler eğitim ve test setlerine ayrılır ve standardize edilir.

Beş regresyon modeli eğitilir: Karar Ağacı Regressor, Random Forest Regressor, Lasso, ElasticNet ve Ridge. Her model için Ortalama Mutlak Hata (MAE), Ortalama Kare Hatası (MSE), Kök Ortalama Kare Hatası (RMSE) ve R-kare (R²) gibi performans metrikleri hesaplanır ve modellerin sigorta maliyeti tahmini yapma başarısı karşılaştırılır.






