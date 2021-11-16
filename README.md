<h1><b>Design, implementation and testing of deep-learning-based driving scenario classifiers for automated vehicles</b></h1>

The aim of this project is to design, implement and test the performance of different neural networks for driving scenarios classification.

The first implemented networks are two versions of a residual 3D convolutional network:
-   the first one (R3D) uses fully 3D kernels
-   the second one (R(2+1)D) separates, in each constituting block, the 2D spatial convolution from the temporal convolution.

Finally, a (YOLOP + CNN) + LSTM stack was realized, which spatially processes each frame only once (instead than every time it appears in a sliding window of scenario analysis), in order to try and obtain latency times compatible with online analysis of driving scenarios.
The networks performance were tests using a synthetic dataset composed by four simple class (approaching static leading vehicle, change lane, following leading vehicle and free ride) in three different conditions (day, night and rain with fog).

<h2><b>Networks’ structure</h2></b>

The three networks are implemented using pytorch and tensorflow:

![R(2+1)D and R3D structure](images/RCN.png?raw=true "Figure 1")

![(YOLOP + CNN) + LSTM structure](images/(YOLOP_+_CNN)_+_LSTM.png?raw=true "Figure 2")

<h2><b>Dataset:</h2></b>

Syntetic dataset obtained using Carla simulator:

![Approaching static leading vehicle](images/Approach.gif?raw=true "Figure 3")
![Change lane](images/changelane.gif?raw=true "Figure 4")
![Following leading vehicle](images/follow.gif?raw=true "Figure 5")
![Free ride](images/freeride.gif?raw=true "Figure 6")

Example of previous dataset's sample processed with YOLOP

![Example of sample od previous dataset processed with YOLOP](images/YOLOP_sample.GIF?raw=true "Figure 7")

<h2><b>Results</h2></b>

The experimental results obtained seem to suggest two different use cases for the analyzed networks: 
-   *Offline*:
	-   R(2+1)D that is more suited when the scenario changes quickly or a low latency is required
	-   R3D that is better for slow-changing scenarios or with low FPS rates.
-   *Online*:
	-   (YOLOP + CNN) + LSTM

<h2><b>Static analysis</b>: Confusion matrix (full video)</h2>

![R(2+1)D confusion matrix (full video)](images/8sec_(2+1)D.png?raw=true "Figure 3") ![R3D confusion matrix (full video)](images/8sec_3D.png?raw=true "Figure 8") ![(YOLOP + CNN) + LSTM confusion matrix (full video)](images/8sec_YOLOP.png?raw=true "Figure 9")

<h2><b>Dinamic analysis</b> (Different length video-clip and conditions)</h2>

![R(2+1)D](images/acc_2+1d.png?raw=true "Figure 6") ![R3D](images/acc3d.png?raw=true "Figure 10") ![(YOLOP + CNN) + LSTM](images/accYOLOP.png?raw=true "Figure 11")

<h2><b>Accuracy and precision performances with different number of testing and training frames</b></h2>

Residual convolutional neural networks tested with 2 seconds video-clip length:

![R(2+1)D and R3D](images/Accuracy_and_Precision.png?raw=true "Figure 12")

(YOLOP + CNN) + LSTM tested with 4 seconds video-clip length:

![(YOLOP + CNN) + LSTM](images/Accuracy_and_Precision_YOLOP.png?raw=true "Figure 13")

<h2>Time inference performances:</h2>

![R(2+1)D time inference](images/tR(2+1)D.png?raw=true "Figure 11") ![R3D time inference](images/tR3D.png?raw=true "Figure 14") ![(YOLOP + CNN) + LSTM time inference](images/tYOLOP.png?raw=true "Figure 15")

<h2>Requirements</h2>

see requirements.txt

<h3><b>Acknowledge</b></h3>

[YOLOP](https://github.com/hustvl/YOLOP)

[R(2+1)D](https://github.com/irhum/R2Plus1D-PyTorch)