# SDStools
A toolbox for analyses of Satellite Derived Shorelines (SDS) generated from CoastSeg (CoastSat)

# RoadMap 
Currently there are some post-processing needs that users of coastseg have requested. Our team has grouped these needs into the following major categories.

### **Interpolation**
- detect nonstationarity
- imputation techniques:
  1) linear interpolation
  2) regression (covariates)
  3) autoregression
  4) ML (covariates)
  5) Deep learning  

### **Uncertainty**
- SDS variability quantification and classification

### **Trend analysis**
- linear trend
- nonlinear trend

### **Classification**
- classify SDS time-series into pre-determined classes
- classify into custom (unsupervised) classes

### **Filtering**
- **Isolating Noise:** Techniques to identify and isolate noise within the shoreline data.
- **Quality:** Tools to assess the quality of the shoreline data.
- **Clouds:** Methods to deal with cloudy data segments and improve accuracy.

### **Time Series/Seasonality Analysis**
- **STL:** Utilize STL (Seasonal-Trend decomposition using LOESS) for shoreline time series decomposition.
- **Seasonal Shorelines:** Analyze the periodic movement of shorelines.
- **Time Series of Narrowest/Widest Beaches:** Assess the evolution of beach widths over time.
- wavelets for nonstationary shoreline timeseries

### **Comparison/Evaluation**
- **Shoreline Location from Situ Surveys (First):** Primary method for ascertaining shoreline location.
- **Shoreline Location with Lidar:** Use of Lidar data for a more granular assessment.

### **Visualization**
- **Animations:** Dynamic representation of shoreline movements.
- **Shorelines Color Coded by Time:** Visualize temporal shoreline changes with color gradients.

### **Other Analysis**
- **Beach Width Time Series (Provide Non-Erodible):** Analyze the temporal changes in beach width.
- **Non-Erodible Line for the Beach:**
- **Shoreline Position Coefficient of Variation:**

