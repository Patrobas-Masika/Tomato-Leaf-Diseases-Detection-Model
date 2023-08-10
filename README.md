<h1>Tomato Leaf Diseases Detection Model</h1>

<h2>Description</h2>
<p align="justify"

The project was designed to create a model for the detection of diseases on tomato leaves. This would involve identifying whether a leaf is healthy or unhealthy. The model was developed using the TensorFlow 2 Object Detection API, which was utilized to train an EfficientDet model. This model was later converted into a TFLite format, enabling it to be deployed on edge devices such as Raspberry Pi. The custom dataset, comprising 4358 images (3488 for training, 435 for validation, and 435 for testing), was obtained from Kaggle. Annotation of the dataset was performed by creating bounding boxes using the LabelImg application program.

To see the inspiration for this project, check out this <a href="https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/Train_TFLite2_Object_Detction_Model.ipynb">one</a>

<b>Note:</b> This description contains snippets of the code in image form. Open the AUGV_Object_Detection_Model.ipynb file (check repository files for it) for the code and to follow through.

</p>
<br/>

<h2>Language Used</h2>

- <b>Python</b> 

<h2>Project Walk-Through:</h2>
<p align="justify"
 
The project was done using Google Colab. If you have no experience with the cloud-based platform, check out this <a href="https://www.youtube.com/watch?v=agj3AxNPDWU&list=PLA83b1JHN4ly56Y7o6vDAT8Szxc3_EdRH">video series</a> for an introduction on how to navigate it. 
</p>

<p align="justify"

As explained earlier, the dataset was annotated using the LabelImg application program. To understand how to use it as well, <a href="https://www.youtube.com/watch?v=fjynQ9P2C08">watch this video</a>. Ensure the images are annotated in PascalVOC format so that for each of the images a .xml file is created. Split the images (and their subsequent .xml files) into Train, Validation, and Test folders. A practical suggestion would be to have 80% of the images in the Train folder and 10% of them in both the Validation and Test folders. Zip them into a folder and name it "Dataset" (without the quotes). Upload the zip folder to the Google Drive associated with the same email address that you are using for Colab. <br><br>
If you need an already prepared dataset for this project, download <a href="https://drive.google.com/file/d/1jPbRL7j4_teEp_mG8tIC-XqJNeN2cWav/view?usp=sharing">Dataset.zip</a>.
</p>

<h3>1. Installing TensorFlow Object Detection Dependencies</h3>
<p align="justify"
 
We first set up the TensorFlow Object Detection API. We'll do this by cloning the TensorFlow models repository and executing a few installation commands. 
<br/>
</p>

<b>Note:</b> This Colab has been set to use TF v2.8.0.
<img src="https://i.imgur.com/DVi73Fr.png"/>
<img src="https://i.imgur.com/Lx69yUO.png"/>
<img src="https://i.imgur.com/mqJ4Pei.png"/>
<img src="https://i.imgur.com/DiCF9QN.png"/>

Warnings or errors related to package dependencies in the preceding code block might arise but you can disregard them.<br><br>
Run the following code block to confirm if everything is working correctly. Confirm there are no errors.

<img src="https://i.imgur.com/eZKNXEL.png"/>

<h3>2. Uploading the Dataset to Colab</h3>

Upload the dataset to Colab

<img src="https://i.imgur.com/hVeEnWW.png"/>

Make a folder on Colab and name it "Dataset" (without the quotes). Unzip the "Dataset.zip" folder into the "Dataset" folder.

<img src="https://i.imgur.com/Z31Cij0.png"/>

<h3>3. Creating Labelmap File and TFRecords</h3>
<p align="justify"
 
In this section, you'll generate a labelmap file for the detector and transform the images into a data file format known as TFRecord, which TensorFlow uses for training. There are existing Python scripts available that facilitate the automatic conversion of data into TFRecord format. <br><br>
First, download and run the data conversion script by running the following section of code.

<img src="https://imgur.com/hL5k0T0.png"/>

Next, create a labelmap.txt file to distinguish between the healthy and unhealthy classes.

<img src="https://live.staticflickr.com/65535/53106500866_7e9127f705_b.jpg"/>

Then, download and upload <a href="https://drive.google.com/file/d/12nsh-EtKV2m5frNECClrVWOw3Az-HZNd/view?usp=sharing">create_csv1.py</a> into your Google Drive folder.<br><br>
Execute the two code segments below. This action will generate TFRecord files for both the training and validation datasets, along with a labelmap.pbtxt file that holds the labelmap in an alternate format.

<img src="https://imgur.com/AzxP2yt.png"/>
<img src="https://imgur.com/JVvs4kV.png"/>
</p>

<h3>4. Setting Up Training Configuration</h3>
<p align="justify">
In this section, we'll establish the model and configure the training settings. We will indicate the specific pre-trained model from the TensorFlow 2 Object Detection Zoo that we are going to use. Each model has an accompanying configuration file which specifies the location, configures training parameters i.e., learning rate and number of training steps, and more. <br><br>
The first code section provides a list for the available models in the TF2 Model Zoo. It also defines the various filenames which are going to be use to download the model and configuration files. For this training, the model used was the efficientdet-d0 model. The second section is used to download the pre-trained model and the configuration file.
<br><br>
<img src="https://imgur.com/rrKebvB.png">
<img src="https://imgur.com/gXVpnC5.png">
<br><br>
Now, proceed by adjusting the configuration file to incorporate the necessary training parameters. In this context, these parameters are the <b>num_steps</b> and <b>batch_size</b>. The <b>num_steps</b> parameter signifies the total number of steps required for model training. The greater the number of steps, the longer the training process will extend. It's important to stay attentive to when the loss curve starts to level off. Once this occurs, you can halt the training, even if all the steps have not been completed.<br>
The <b>batch_size</b> parameter indicates the quantity of images utilized in each individual training step. The larger the batch size, the fewer overall steps the training will encompass. However, this size is constrained by the available GPU memory. In this specific instance, considering the project's execution on Google Colab, a batch size of 4 would be optimal for efficientdet-d0 models.<br><br>
<img src="https://imgur.com/rM9vMPC.png">
<br><br>
Next, set the location of the pretrained model file, the config file, and total number of classes.
<br><br>
<img src="https://imgur.com/61yB3Kg.png">
<br><br>
Moving forward, rewrite the configuration file to incorporate the training parameters you've recently defined. The subsequent code segment will automatically substitute the required parameters in the downloaded .config file and then save it as a customized "pipeline_file.config" file.
<br><br>
<img src="https://imgur.com/Lu4XWaQ.png">
<img src="https://imgur.com/Zz0BCpq.png">
<br><br>
If you want, you can run the following section to display the contents of the configuration file.
<br><br>
<img src="https://imgur.com/El6dHq0.png">
<br><br>
Now, set the locations of the configuration file and model output directory as variables so that you can be able to reference them later when you call the training command.
<br><br>
<img src="https://imgur.com/UKKo8q4.png">
<br><br>
</p>

<h3>5. Training TFLite Detection Model</h3>
<p align="justify">
The moment has arrived to commence training your model. For monitoring the training progress, initiate a TensorBoard session using the following code snippet. The display will become active only once training has initiated. Once you've initiated the training process, return and click the refresh button on the TensorBoard interface.<br><br>
<b>Note:</b> The TensorBoard exhibits log messages every 100 steps. Consequently, please allow it some time before it begins to showcase content.
<br><br>
<img src="https://imgur.com/JD3oAqw.png">
<br><br>
</p>



